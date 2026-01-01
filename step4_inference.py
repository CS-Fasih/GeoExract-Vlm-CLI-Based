"""
Step 4: Inference with Chain-of-Thought Reasoning
"Thinking Model" Implementation for Satellite Imagery Analysis
"""

import os
import torch
from PIL import Image
from typing import Optional, List, Dict, Any
import json

# ============================================
# CONFIGURATION
# ============================================

# Model paths
MODEL_PATH = "/content/qwen2vl_finetuned"
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# Chain-of-Thought System Prompt
COT_SYSTEM_PROMPT = """You are an expert geospatial analyst specializing in satellite imagery interpretation. 

Before providing your final answer, you MUST follow this structured analytical framework:

## ANALYSIS PROTOCOL:

### Step 1: Initial Overview
- Identify the type of imagery (urban, rural, coastal, etc.)
- Note the approximate scale and coverage area
- Assess image quality and visibility conditions

### Step 2: Structure Detection
- Scan systematically from top-left to bottom-right
- Identify and count distinct buildings/structures
- Classify structures by type (residential, commercial, industrial)
- Note building sizes, shapes, and roof characteristics

### Step 3: Infrastructure Analysis
- Identify road networks and transportation corridors
- Locate waterways, canals, or coastal features
- Note parking areas, open spaces, and vegetation

### Step 4: Density Assessment
- Calculate approximate building density
- Classify as low/medium/high density
- Compare to typical urban patterns

### Step 5: Final Synthesis
- Summarize key findings
- Provide confidence level for your assessment
- Answer the specific question asked

---
Now analyze the image and answer the question using this framework."""

# Alternative shorter prompt for faster inference
SHORT_COT_PROMPT = """You are a geospatial analyst. Before answering, analyze the image step-by-step:
1. Identify terrain type and land coverage
2. Count and classify visible structures  
3. Assess urban density and infrastructure
4. Note distinctive features
Then provide your final answer with confidence level."""

# ============================================
# MODEL LOADING
# ============================================

class SatelliteVLM:
    """Vision-Language Model for Satellite Imagery Analysis"""
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        base_model: str = BASE_MODEL,
        use_4bit: bool = True,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.use_4bit = use_4bit
        self.device = device
        
        self.model = None
        self.processor = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model"""
        print("🔧 Loading Satellite VLM...")
        
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig
        )
        from peft import PeftModel
        
        # Check if we have a fine-tuned model
        if os.path.exists(self.model_path):
            print(f"📥 Loading fine-tuned model from {self.model_path}")
            
            # Quantization config
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            else:
                bnb_config = None
            
            # Load base model
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Load LoRA adapters
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.model_path
                )
                print("✅ Loaded LoRA adapters")
            except Exception as e:
                print(f"⚠️ Could not load LoRA adapters: {e}")
                print("Using base model instead")
                self.model = base_model
            
            # Load processor
            if os.path.exists(os.path.join(self.model_path, "processor_config.json")):
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            else:
                self.processor = AutoProcessor.from_pretrained(self.base_model)
        
        else:
            print(f"📥 Loading base model: {self.base_model}")
            
            # Quantization config
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            else:
                bnb_config = None
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"✅ Model loaded! Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def analyze(
        self,
        image_path: str,
        question: str,
        use_cot: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Analyze a satellite image with Chain-of-Thought reasoning
        
        Args:
            image_path: Path to the satellite image
            question: Question about the image
            use_cot: Whether to use Chain-of-Thought prompting
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Dictionary containing analysis results
        """
        print(f"🔍 Analyzing: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize for memory efficiency
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Build conversation with CoT prompt
        system_prompt = COT_SYSTEM_PROMPT if use_cot else SHORT_COT_PROMPT
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode response
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # Parse response into thinking and answer
        result = self._parse_response(response)
        result['image_path'] = image_path
        result['question'] = question
        
        return result
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse CoT response into thinking steps and final answer"""
        
        # Try to extract structured sections
        sections = {
            'initial_overview': '',
            'structure_detection': '',
            'infrastructure_analysis': '',
            'density_assessment': '',
            'final_synthesis': '',
            'raw_response': response
        }
        
        # Simple parsing based on step markers
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            if 'step 1' in line_lower or 'initial overview' in line_lower:
                current_section = 'initial_overview'
            elif 'step 2' in line_lower or 'structure' in line_lower:
                current_section = 'structure_detection'
            elif 'step 3' in line_lower or 'infrastructure' in line_lower:
                current_section = 'infrastructure_analysis'
            elif 'step 4' in line_lower or 'density' in line_lower:
                current_section = 'density_assessment'
            elif 'step 5' in line_lower or 'final' in line_lower or 'synthesis' in line_lower:
                current_section = 'final_synthesis'
            
            if current_section:
                sections[current_section] += line + '\n'
        
        # Extract final answer (usually after "Final" or at the end)
        if sections['final_synthesis']:
            sections['answer'] = sections['final_synthesis']
        else:
            # Take last paragraph as answer
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            sections['answer'] = paragraphs[-1] if paragraphs else response
        
        # Extract thinking process
        sections['thinking'] = '\n'.join([
            sections.get('initial_overview', ''),
            sections.get('structure_detection', ''),
            sections.get('infrastructure_analysis', ''),
            sections.get('density_assessment', '')
        ]).strip()
        
        return sections
    
    def batch_analyze(
        self,
        image_paths: List[str],
        question: str,
        use_cot: bool = True
    ) -> List[Dict[str, Any]]:
        """Analyze multiple images with the same question"""
        results = []
        for img_path in image_paths:
            try:
                result = self.analyze(img_path, question, use_cot)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'question': question,
                    'error': str(e)
                })
        return results

# ============================================
# INTERACTIVE DEMO
# ============================================

def interactive_demo():
    """Interactive demo for satellite image analysis"""
    print("=" * 60)
    print("🛰️ Satellite Image Analysis - Interactive Demo")
    print("=" * 60)
    
    # Initialize model
    vlm = SatelliteVLM()
    
    # Demo questions
    demo_questions = [
        "How many buildings can you identify in this satellite image?",
        "Describe the urban density in this area.",
        "What infrastructure elements are visible?",
        "Is this primarily residential or commercial area?",
        "Describe the overall layout of this satellite view."
    ]
    
    print("\n📋 Sample questions you can ask:")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-" * 60)
    print("Enter 'quit' to exit")
    print("-" * 60)
    
    while True:
        # Get image path
        image_path = input("\n📷 Enter image path (or 'quit'): ").strip()
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        # Get question
        question = input("❓ Enter your question: ").strip()
        if not question:
            question = demo_questions[0]
            print(f"Using default question: {question}")
        
        # Analyze
        print("\n🔍 Analyzing (this may take a moment)...\n")
        result = vlm.analyze(image_path, question, use_cot=True)
        
        # Display results
        print("=" * 60)
        print("📊 ANALYSIS RESULTS")
        print("=" * 60)
        
        if result.get('thinking'):
            print("\n🧠 THINKING PROCESS:")
            print("-" * 40)
            print(result['thinking'][:500] + "..." if len(result.get('thinking', '')) > 500 else result.get('thinking', 'N/A'))
        
        print("\n✅ FINAL ANSWER:")
        print("-" * 40)
        print(result.get('answer', result.get('raw_response', 'No response')))
        
        print("\n" + "=" * 60)

# ============================================
# BATCH PROCESSING
# ============================================

def process_directory(
    image_dir: str,
    output_file: str,
    question: str = "Describe the buildings and infrastructure in this satellite image."
):
    """Process all images in a directory"""
    print(f"📂 Processing images from: {image_dir}")
    
    # Initialize model
    vlm = SatelliteVLM()
    
    # Get image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif']
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    results = []
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
        
        try:
            result = vlm.analyze(img_path, question, use_cot=True)
            results.append(result)
            
            # Print summary
            print(f"  ✅ Buildings mentioned: {'yes' if 'building' in result.get('raw_response', '').lower() else 'no'}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'image_path': img_path,
                'error': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    return results

# ============================================
# MAIN
# ============================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Satellite Image VLM Inference")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--question", type=str, default="Describe the buildings in this satellite image.")
    parser.add_argument("--batch-dir", type=str, help="Directory for batch processing")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for batch processing")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--no-cot", action="store_true", help="Disable Chain-of-Thought")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    elif args.batch_dir:
        process_directory(args.batch_dir, args.output, args.question)
    elif args.image:
        vlm = SatelliteVLM()
        result = vlm.analyze(args.image, args.question, use_cot=not args.no_cot)
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS RESULT")
        print("=" * 60)
        print(result.get('raw_response', 'No response'))
    else:
        # Run quick demo
        print("Run with --interactive for interactive mode")
        print("Run with --image <path> for single image analysis")
        print("Run with --batch-dir <dir> for batch processing")
        
        # Quick test with base model
        print("\n🔧 Quick model test...")
        vlm = SatelliteVLM()
        print("✅ Model loaded successfully!")

if __name__ == "__main__":
    main()
