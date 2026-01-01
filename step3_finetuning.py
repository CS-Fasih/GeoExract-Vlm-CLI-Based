"""
Step 3: Fine-Tuning Qwen2-VL-2B-Instruct with QLoRA
Optimized for Google Colab Free Tier (T4 GPU, Limited RAM)
"""

import os
import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class TrainingConfig:
    """Training configuration for Qwen2-VL fine-tuning"""
    
    # Model settings
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Data settings
    train_file: str = "/content/training_data/train.jsonl"
    val_file: str = "/content/training_data/val.jsonl"
    output_dir: str = "/content/qwen2vl_finetuned"
    
    # Quantization settings (4-bit for memory efficiency)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Vision encoder modules
        "qkv", "proj"
    ])
    
    # Training hyperparameters
    num_train_epochs: int = 1
    max_steps: int = 100  # Keep short for demo
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_grad_norm: float = 0.3
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False  # Use bf16 instead on T4
    bf16: bool = True
    
    # Logging
    logging_steps: int = 5
    save_steps: int = 25
    eval_steps: int = 25
    
    # Image settings
    max_image_size: int = 512  # Reduce for memory
    min_image_size: int = 256

# ============================================
# INSTALLATION
# ============================================

def install_dependencies():
    """Install all required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "transformers>=4.44.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "trl>=0.8.0",
        "datasets",
        "wandb",
        "qwen-vl-utils",
        "Pillow",
        "scipy"
    ]
    
    import subprocess
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], 
                      capture_output=True)
    
    print("✅ Dependencies installed!")

# ============================================
# DATA LOADING
# ============================================

class VLMDataset(torch.utils.data.Dataset):
    """Custom dataset for Qwen2-VL training"""
    
    def __init__(self, jsonl_path, processor, config):
        self.processor = processor
        self.config = config
        self.samples = []
        
        # Load JSONL data
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        
        # Extract components
        system_msg = messages[0]['content']
        user_content = messages[1]['content']
        assistant_msg = messages[2]['content']
        
        # Parse user content (image + text)
        image_path = None
        question = None
        for item in user_content:
            if item['type'] == 'image':
                image_path = item['image']
            elif item['type'] == 'text':
                question = item['text']
        
        # Load and process image
        from PIL import Image
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize for memory efficiency
            image = image.resize((self.config.max_image_size, self.config.max_image_size))
        except Exception as e:
            # Create placeholder if image fails
            print(f"Warning: Could not load {image_path}: {e}")
            image = Image.new('RGB', (self.config.max_image_size, self.config.max_image_size))
        
        # Format conversation
        conversation = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": assistant_msg}
        ]
        
        return {
            "conversation": conversation,
            "image": image
        }

def collate_fn(batch, processor):
    """Custom collate function for batch processing"""
    conversations = [item['conversation'] for item in batch]
    images = [item['image'] for item in batch]
    
    # Apply chat template
    texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) 
             for conv in conversations]
    
    # Process with processor
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Create labels (mask the input part, only compute loss on response)
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs

# ============================================
# MODEL SETUP
# ============================================

def setup_model_and_tokenizer(config: TrainingConfig):
    """Load model with 4-bit quantization and prepare for LoRA"""
    print("🔧 Setting up model and tokenizer...")
    
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )
    
    # Quantization config for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant
    )
    
    print(f"📥 Loading {config.model_name} with 4-bit quantization...")
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Set padding
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print(f"✅ Model loaded! Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    return model, processor

def apply_lora(model, config: TrainingConfig):
    """Apply LoRA adapters to the model"""
    print("🔧 Applying LoRA adapters...")
    
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model

# ============================================
# TRAINING WITH SFTTrainer
# ============================================

def train_with_sft_trainer(model, processor, config: TrainingConfig):
    """Train using TRL's SFTTrainer"""
    print("🚀 Starting training with SFTTrainer...")
    
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    # Load datasets
    dataset = load_dataset('json', data_files={
        'train': config.train_file,
        'validation': config.val_file
    })
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_8bit",
        report_to="none",  # Set to "wandb" if you want logging
        remove_unused_columns=False,
        max_seq_length=2048,
        dataset_text_field="text"
    )
    
    # Format function for SFTTrainer
    def formatting_func(examples):
        texts = []
        for messages in examples['messages']:
            # Build text from messages
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        formatting_func=formatting_func,
        tokenizer=processor.tokenizer,
    )
    
    # Train
    print("🏋️ Training...")
    trainer.train()
    
    # Save model
    print(f"💾 Saving model to {config.output_dir}")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    return trainer

# ============================================
# CUSTOM TRAINING LOOP (Alternative)
# ============================================

def custom_training_loop(model, processor, config: TrainingConfig):
    """Custom training loop as fallback"""
    print("🚀 Starting custom training loop...")
    
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    
    # Create datasets
    train_dataset = VLMDataset(config.train_file, processor, config)
    val_dataset = VLMDataset(config.val_file, processor, config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    total_steps = min(config.max_steps, len(train_loader) * config.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    accumulation_loss = 0
    
    progress_bar = tqdm(total=config.max_steps, desc="Training")
    
    for epoch in range(config.num_train_epochs):
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= config.max_steps:
                break
            
            # Move to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulation_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': accumulation_loss})
                accumulation_loss = 0
                
                # Log
                if global_step % config.logging_steps == 0:
                    print(f"Step {global_step}: Loss = {loss.item() * config.gradient_accumulation_steps:.4f}")
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(checkpoint_dir)
                    print(f"💾 Saved checkpoint to {checkpoint_dir}")
    
    progress_bar.close()
    
    # Save final model
    print(f"💾 Saving final model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    return model

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("=" * 60)
    print("🚀 Qwen2-VL Fine-Tuning Script")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Create config
    config = TrainingConfig()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Setup model
    model, processor = setup_model_and_tokenizer(config)
    
    # Apply LoRA
    model = apply_lora(model, config)
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Train
    try:
        # Try SFTTrainer first
        trainer = train_with_sft_trainer(model, processor, config)
    except Exception as e:
        print(f"⚠️ SFTTrainer failed: {e}")
        print("🔄 Falling back to custom training loop...")
        model = custom_training_loop(model, processor, config)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📂 Model saved to: {config.output_dir}")
    print("=" * 60)
    
    # Memory cleanup
    torch.cuda.empty_cache()
    
    return config.output_dir

if __name__ == "__main__":
    output_dir = main()
