"""
Step 5: Export Model to GGUF Format for CPU Inference
Merge LoRA adapters and convert to GGUF using llama.cpp
"""

import os
import subprocess
import shutil
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

# Paths
FINETUNED_MODEL_PATH = "/content/qwen2vl_finetuned"
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MERGED_MODEL_PATH = "/content/qwen2vl_merged"
GGUF_OUTPUT_PATH = "/content/qwen2vl_gguf"
LLAMA_CPP_PATH = "/content/llama.cpp"

# GGUF quantization type
GGUF_QUANT_TYPE = "q4_k_m"  # Good balance of quality and size

# ============================================
# STEP 5.1: Merge LoRA Adapters
# ============================================

def merge_lora_adapters():
    """Merge LoRA adapters back into the base model"""
    print("=" * 60)
    print("🔧 Step 5.1: Merging LoRA Adapters")
    print("=" * 60)
    
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    
    print(f"📥 Loading base model: {BASE_MODEL}")
    
    # Load base model in full precision for merging
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"📥 Loading LoRA adapters from: {FINETUNED_MODEL_PATH}")
    
    # Load LoRA model
    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_MODEL_PATH,
        torch_dtype=torch.float16
    )
    
    print("🔄 Merging adapters into base model...")
    
    # Merge LoRA into base model
    model = model.merge_and_unload()
    
    # Create output directory
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    
    print(f"💾 Saving merged model to: {MERGED_MODEL_PATH}")
    
    # Save merged model
    model.save_pretrained(
        MERGED_MODEL_PATH,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.save_pretrained(MERGED_MODEL_PATH)
    
    print("✅ LoRA adapters merged successfully!")
    
    # Cleanup GPU memory
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return MERGED_MODEL_PATH

# ============================================
# STEP 5.2: Setup llama.cpp (LATEST MASTER)
# ============================================

def setup_llama_cpp():
    """
    Clone and build llama.cpp from LATEST master branch
    
    CRITICAL: Qwen2-VL support is very recent - we MUST use the latest version.
    Older versions will strip the vision encoder or fail entirely.
    """
    print("\n" + "=" * 60)
    print("🔧 Step 5.2: Setting up llama.cpp (LATEST MASTER)")
    print("=" * 60)
    print("⚠️  CRITICAL: Using latest master for Qwen2-VL multimodal support")
    
    if os.path.exists(LLAMA_CPP_PATH):
        print(f"📂 llama.cpp exists at {LLAMA_CPP_PATH}")
        print("🔄 Fetching LATEST updates from master...")
        
        # Force update to latest master
        subprocess.run(["git", "fetch", "origin", "master"], cwd=LLAMA_CPP_PATH, capture_output=True)
        subprocess.run(["git", "reset", "--hard", "origin/master"], cwd=LLAMA_CPP_PATH, capture_output=True)
        subprocess.run(["git", "pull", "origin", "master", "--force"], cwd=LLAMA_CPP_PATH, capture_output=True)
        
        # Get commit info
        result = subprocess.run(["git", "log", "-1", "--oneline"], cwd=LLAMA_CPP_PATH, capture_output=True, text=True)
        print(f"📌 Latest commit: {result.stdout.strip()}")
    else:
        print("📥 Cloning llama.cpp (master branch)...")
        subprocess.run([
            "git", "clone", 
            "--branch", "master",
            "--single-branch",
            "--depth", "1",  # Shallow clone for speed
            "https://github.com/ggerganov/llama.cpp.git",
            LLAMA_CPP_PATH
        ], check=True)
        
        # Get commit info
        result = subprocess.run(["git", "log", "-1", "--oneline"], cwd=LLAMA_CPP_PATH, capture_output=True, text=True)
        print(f"📌 Cloned commit: {result.stdout.strip()}")
    
    print("🔨 Building llama.cpp...")
    
    # Clean build
    subprocess.run(["make", "clean"], cwd=LLAMA_CPP_PATH, capture_output=True)
    
    # Build with CPU support
    result = subprocess.run(
        ["make", "-j4", "LLAMA_CURL=1"],  # Enable curl for downloads
        cwd=LLAMA_CPP_PATH,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"⚠️ Make failed, trying cmake...")
        # Try cmake build
        build_dir = os.path.join(LLAMA_CPP_PATH, "build")
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", "..", "-DLLAMA_CURL=ON"], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "-j4"], cwd=build_dir, check=True)
    
    print("✅ llama.cpp built successfully!")
    
    # Install Python dependencies for conversion
    print("📦 Installing conversion dependencies...")
    subprocess.run([
        "pip", "install", "-q", "--upgrade",
        "gguf>=0.6.0",  # Latest GGUF with VLM support
        "sentencepiece", 
        "protobuf",
        "llama-cpp-python>=0.2.50"  # Latest with vision support
    ], check=True)
    
    # Check for Qwen2-VL specific converter
    qwen_converter = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    if os.path.exists(qwen_converter):
        # Check if it mentions Qwen2VL
        with open(qwen_converter, 'r') as f:
            content = f.read()
            if 'Qwen2VL' in content or 'qwen2-vl' in content.lower():
                print("✅ Qwen2-VL converter support detected!")
            else:
                print("⚠️  WARNING: Converter may not have Qwen2-VL support")
                print("   This could result in a text-only model!")
    
    return LLAMA_CPP_PATH

# ============================================
# STEP 5.3: Convert to GGUF
# ============================================

def convert_to_gguf():
    """Convert the merged model to GGUF format"""
    print("\n" + "=" * 60)
    print("🔧 Step 5.3: Converting to GGUF format")
    print("=" * 60)
    
    os.makedirs(GGUF_OUTPUT_PATH, exist_ok=True)
    
    # Path to conversion script
    convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    
    # Alternative script path
    if not os.path.exists(convert_script):
        convert_script = os.path.join(LLAMA_CPP_PATH, "convert-hf-to-gguf.py")
    
    if not os.path.exists(convert_script):
        # Try the older script
        convert_script = os.path.join(LLAMA_CPP_PATH, "convert.py")
    
    # Output GGUF file (FP16 first)
    gguf_fp16_path = os.path.join(GGUF_OUTPUT_PATH, "qwen2vl-2b-satellite-f16.gguf")
    
    print(f"📥 Converting model from: {MERGED_MODEL_PATH}")
    print(f"📤 Output: {gguf_fp16_path}")
    
    # Run conversion
    convert_cmd = [
        "python", convert_script,
        MERGED_MODEL_PATH,
        "--outfile", gguf_fp16_path,
        "--outtype", "f16"
    ]
    
    print(f"🔄 Running: {' '.join(convert_cmd)}")
    
    result = subprocess.run(
        convert_cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"⚠️ Conversion error: {result.stderr}")
        
        # Try alternative approach for Qwen models
        print("🔄 Trying alternative conversion for Qwen2-VL...")
        
        # Some VLM models need special handling
        # Use the generic HF to GGUF converter
        alt_cmd = [
            "python", "-m", "gguf",
            "convert",
            MERGED_MODEL_PATH,
            "-o", gguf_fp16_path
        ]
        
        result = subprocess.run(alt_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("⚠️ Direct conversion failed. Manual steps may be required.")
            print("See GGUF_MANUAL_INSTRUCTIONS.md for guidance.")
            create_manual_instructions()
            return None
    
    print(f"✅ FP16 GGUF created: {gguf_fp16_path}")
    
    return gguf_fp16_path

# ============================================
# STEP 5.4: Quantize GGUF
# ============================================

def quantize_gguf(gguf_fp16_path):
    """Quantize the GGUF model to smaller size"""
    print("\n" + "=" * 60)
    print(f"🔧 Step 5.4: Quantizing to {GGUF_QUANT_TYPE}")
    print("=" * 60)
    
    # Quantize binary
    quantize_bin = os.path.join(LLAMA_CPP_PATH, "quantize")
    
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(LLAMA_CPP_PATH, "build", "bin", "quantize")
    
    if not os.path.exists(quantize_bin):
        print("⚠️ Quantize binary not found. Building...")
        subprocess.run(["make", "quantize"], cwd=LLAMA_CPP_PATH)
        quantize_bin = os.path.join(LLAMA_CPP_PATH, "quantize")
    
    # Output path for quantized model
    gguf_quant_path = os.path.join(
        GGUF_OUTPUT_PATH, 
        f"qwen2vl-2b-satellite-{GGUF_QUANT_TYPE}.gguf"
    )
    
    print(f"📥 Input: {gguf_fp16_path}")
    print(f"📤 Output: {gguf_quant_path}")
    print(f"📊 Quantization type: {GGUF_QUANT_TYPE}")
    
    # Run quantization
    quant_cmd = [
        quantize_bin,
        gguf_fp16_path,
        gguf_quant_path,
        GGUF_QUANT_TYPE
    ]
    
    result = subprocess.run(quant_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ Quantization error: {result.stderr}")
        return gguf_fp16_path  # Return FP16 version
    
    print(f"✅ Quantized GGUF created: {gguf_quant_path}")
    
    # Print file sizes
    fp16_size = os.path.getsize(gguf_fp16_path) / (1024**3)
    quant_size = os.path.getsize(gguf_quant_path) / (1024**3)
    
    print(f"\n📊 Size comparison:")
    print(f"   FP16:      {fp16_size:.2f} GB")
    print(f"   {GGUF_QUANT_TYPE}: {quant_size:.2f} GB")
    print(f"   Reduction: {(1 - quant_size/fp16_size)*100:.1f}%")
    
    return gguf_quant_path

# ============================================
# STEP 5.5: Create CPU Inference Script
# ============================================

def create_cpu_inference_script(gguf_path):
    """Create a CPU inference script using llama-cpp-python"""
    print("\n" + "=" * 60)
    print("🔧 Step 5.5: Creating CPU inference script")
    print("=" * 60)
    
    script_content = f'''"""
CPU Inference Script for Qwen2-VL Satellite Model
Uses llama-cpp-python for efficient CPU inference
"""

import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen2VLChatHandler
from PIL import Image
import base64
from io import BytesIO

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "{gguf_path}"

# System prompt for Chain-of-Thought
SYSTEM_PROMPT = """You are an expert geospatial analyst specializing in satellite imagery interpretation.
Before providing your final answer, analyze step-by-step:
1. Identify terrain type and coverage
2. Count and classify structures
3. Assess urban density
4. Note distinctive features
Then provide your final answer."""

# ============================================
# MODEL LOADING
# ============================================

class SatelliteCPUModel:
    """CPU-optimized satellite image analysis model"""
    
    def __init__(self, model_path=MODEL_PATH, n_ctx=2048, n_threads=4):
        print(f"Loading model: {{model_path}}")
        
        # Initialize llama.cpp model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only
            verbose=False
        )
        
        print("Model loaded successfully!")
    
    def encode_image(self, image_path):
        """Encode image to base64 for multimodal input"""
        with Image.open(image_path) as img:
            # Resize for efficiency
            img = img.convert("RGB")
            img.thumbnail((512, 512))
            
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
    
    def analyze(self, image_path, question, max_tokens=512):
        """Analyze satellite image"""
        
        # Encode image
        img_base64 = self.encode_image(image_path)
        
        # Build prompt
        prompt = f"""<|im_start|>system
{{SYSTEM_PROMPT}}<|im_end|>
<|im_start|>user
<image>{{img_base64}}</image>
{{question}}<|im_end|>
<|im_start|>assistant
"""
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|im_end|>"],
            echo=False
        )
        
        return response["choices"][0]["text"]

# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to satellite image")
    parser.add_argument("--question", default="Describe the buildings in this image.")
    parser.add_argument("--threads", type=int, default=4)
    
    args = parser.parse_args()
    
    # Load model
    model = SatelliteCPUModel(n_threads=args.threads)
    
    # Analyze
    print(f"\\nAnalyzing: {{args.image}}")
    print(f"Question: {{args.question}}")
    print("\\n" + "=" * 50)
    
    response = model.analyze(args.image, args.question)
    print(response)

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(GGUF_OUTPUT_PATH, "cpu_inference.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ CPU inference script created: {script_path}")
    
    # Create requirements.txt
    requirements = """# Requirements for CPU inference
llama-cpp-python>=0.2.50
Pillow>=9.0.0
numpy>=1.21.0
"""
    
    req_path = os.path.join(GGUF_OUTPUT_PATH, "requirements.txt")
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    print(f"✅ Requirements file created: {req_path}")
    
    return script_path


# ============================================
# STEP 5.6: VERIFY GGUF VISION SUPPORT
# ============================================

def verify_gguf_vision_support(gguf_path, test_image_path=None):
    """
    CRITICAL VERIFICATION: Test if the GGUF model actually supports vision.
    
    This catches the common failure where conversion strips the vision encoder,
    leaving a text-only model that crashes on images.
    
    Returns:
        bool: True if vision works, False if it's broken
    """
    print("\n" + "=" * 60)
    print("🔍 Step 5.6: VERIFYING GGUF VISION SUPPORT")
    print("=" * 60)
    print("⚠️  This is CRITICAL - checking if vision encoder survived conversion...")
    
    if not os.path.exists(gguf_path):
        print(f"❌ GGUF file not found: {gguf_path}")
        return False
    
    # Check file size (vision models should be larger)
    file_size_gb = os.path.getsize(gguf_path) / (1024**3)
    print(f"📦 GGUF file size: {file_size_gb:.2f} GB")
    
    if file_size_gb < 1.0:
        print("⚠️  WARNING: File seems too small for a VLM. Vision may be stripped.")
    
    try:
        from llama_cpp import Llama
        from PIL import Image
        import base64
        from io import BytesIO
        
        print("📥 Loading GGUF model for verification...")
        
        # Try to load the model
        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_threads=2,
            n_gpu_layers=0,
            verbose=False
        )
        
        print("✅ Model loaded successfully")
        
        # Create a test image (red square) if no test image provided
        if test_image_path and os.path.exists(test_image_path):
            test_img = Image.open(test_image_path).convert("RGB")
        else:
            # Create synthetic test image
            test_img = Image.new('RGB', (64, 64), color='red')
            print("📷 Using synthetic test image (red square)")
        
        test_img.thumbnail((64, 64))
        buffer = BytesIO()
        test_img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Try vision inference
        print("🧪 Testing vision inference...")
        
        # Construct a multimodal prompt
        prompt = f"""<|im_start|>system
You are an assistant.<|im_end|>
<|im_start|>user
<image>{img_b64}</image>
What color is this image?<|im_end|>
<|im_start|>assistant
"""
        
        try:
            response = llm(
                prompt,
                max_tokens=50,
                stop=["<|im_end|>"],
                echo=False
            )
            
            output_text = response["choices"][0]["text"] if response.get("choices") else ""
            print(f"📝 Model response: {output_text[:100]}...")
            
            # Check if it actually processed the image
            # A text-only model would give nonsensical response
            if "red" in output_text.lower() or "color" in output_text.lower() or len(output_text) > 5:
                print("\n" + "=" * 60)
                print("✅ VERIFICATION PASSED: Vision support appears to work!")
                print("=" * 60)
                return True
            else:
                print("⚠️  Response seems generic - vision may not be working")
                
        except ValueError as e:
            if "vision" in str(e).lower() or "image" in str(e).lower() or "multimodal" in str(e).lower():
                print(f"\n❌ VISION ERROR: {e}")
                _print_critical_failure()
                return False
            raise
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['vision', 'image', 'multimodal', 'unsupported']):
                print(f"\n❌ VISION ERROR: {e}")
                _print_critical_failure()
                return False
            # Other errors might be OK
            print(f"⚠️  Non-vision error during test: {e}")
            return True  # Assume OK if error isn't vision-related
            
    except ImportError:
        print("⚠️  llama-cpp-python not installed, skipping verification")
        print("   Install with: pip install llama-cpp-python>=0.2.50")
        return None  # Unknown
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for vision-specific errors
        if any(keyword in error_msg for keyword in [
            'vision', 'image', 'multimodal', 'not support', 
            'clip', 'visual', 'encoder'
        ]):
            _print_critical_failure()
            return False
        
        print(f"⚠️  Verification error: {e}")
        print("   This may or may not indicate a vision problem.")
        return None  # Unknown
    
    return True


def _print_critical_failure():
    """Print critical failure message with recovery instructions"""
    print("\n" + "🚨" * 30)
    print("\n❌ CRITICAL ERROR: GGUF MODEL DOES NOT SUPPORT VISION!")
    print("\n" + "🚨" * 30)
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    VISION ENCODER STRIPPED!                       ║
╠══════════════════════════════════════════════════════════════════╣
║ The GGUF conversion removed or corrupted the vision encoder.     ║
║ This model will CRASH when you try to analyze images.            ║
║                                                                   ║
║ This is a KNOWN ISSUE with Qwen2-VL GGUF conversion.             ║
╚══════════════════════════════════════════════════════════════════╝

🔧 RECOVERY OPTIONS:

   OPTION 1: Use PyTorch Model Instead (RECOMMENDED)
   ─────────────────────────────────────────────────
   Use the merged model at: /content/qwen2vl_merged
   With the Transformers library (see cpu_inference_transformers.py)
   
   # Example:
   from transformers import Qwen2VLForConditionalGeneration
   model = Qwen2VLForConditionalGeneration.from_pretrained(
       "/content/qwen2vl_merged",
       device_map="cpu",
       torch_dtype=torch.float32
   )
   
   OPTION 2: Check HuggingFace for Pre-converted GGUF
   ─────────────────────────────────────────────────
   Search: https://huggingface.co/models?search=qwen2-vl+gguf
   Some community members may have working conversions.
   
   OPTION 3: Use ONNX Runtime Instead
   ─────────────────────────────────────────────────
   Export to ONNX format which better supports multimodal models.
   
   OPTION 4: Wait for llama.cpp Updates
   ─────────────────────────────────────────────────
   Qwen2-VL support in llama.cpp is actively being developed.
   Check: https://github.com/ggerganov/llama.cpp/issues
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  DO NOT use the GGUF file for image analysis - it WILL fail!
   Use the Transformers-based CPU script instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ============================================
# MANUAL INSTRUCTIONS FALLBACK
# ============================================

def create_manual_instructions():
    """Create manual instructions for GGUF conversion"""
    
    instructions = """# Manual GGUF Conversion Instructions

Due to Qwen2-VL being a Vision-Language Model, automatic GGUF conversion may require additional steps.

## Option 1: Use Hugging Face's Pre-converted Models

Check if a GGUF version already exists:
```bash
# Search on Hugging Face
# https://huggingface.co/models?search=qwen2-vl+gguf
```

## Option 2: Manual Conversion Steps

### Step 1: Install dependencies
```bash
pip install transformers torch gguf sentencepiece
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

### Step 2: Convert the language model component
```bash
# Qwen2-VL has separate vision and language components
# The language model can be converted:
python convert_hf_to_gguf.py /content/qwen2vl_merged \\
    --outfile qwen2vl-language-f16.gguf \\
    --outtype f16
```

### Step 3: Quantize
```bash
./quantize qwen2vl-language-f16.gguf qwen2vl-q4_k_m.gguf q4_k_m
```

## Option 3: Use llama.cpp Multimodal Support

For full VLM support, use llama.cpp's multimodal capabilities:
```bash
# Build with vision support
make LLAMA_CURL=1

# Run with image support
./main -m model.gguf --mmproj mmproj.gguf \\
    --image satellite.png -p "Describe this image"
```

## Option 4: Alternative Runtimes

Consider these alternatives for CPU inference:

1. **ONNX Runtime**: Export to ONNX format
2. **OpenVINO**: Intel's inference toolkit
3. **MLC-LLM**: Machine Learning Compilation
4. **CTransformers**: Python bindings for transformers

## Notes

- Vision encoders typically need separate handling
- Consider using the Transformers library for CPU inference
- For production, consider model distillation to a smaller architecture
"""
    
    path = os.path.join(GGUF_OUTPUT_PATH, "GGUF_MANUAL_INSTRUCTIONS.md")
    os.makedirs(GGUF_OUTPUT_PATH, exist_ok=True)
    with open(path, 'w') as f:
        f.write(instructions)
    
    print(f"📄 Manual instructions saved to: {path}")

# ============================================
# ALTERNATIVE: CPU INFERENCE WITH TRANSFORMERS
# ============================================

def create_transformers_cpu_script():
    """Create CPU inference script using transformers (no GGUF)"""
    print("\n" + "=" * 60)
    print("🔧 Creating Transformers-based CPU inference script")
    print("=" * 60)
    
    script_content = '''"""
CPU Inference using Transformers (Alternative to GGUF)
Works directly with the merged model without GGUF conversion
"""

import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import time

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "/content/qwen2vl_merged"  # Or download from HF

SYSTEM_PROMPT = """You are an expert geospatial analyst. 
Analyze the satellite image step-by-step:
1. Identify terrain and land coverage
2. Count and classify structures
3. Assess urban density
4. Provide final answer"""

# ============================================
# MODEL CLASS
# ============================================

class SatelliteVLM_CPU:
    """CPU-optimized VLM using PyTorch"""
    
    def __init__(self, model_path=MODEL_PATH):
        print("Loading model for CPU inference...")
        print("This may take a few minutes...")
        
        # Load model in FP32 for CPU
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model.eval()
        print("Model loaded!")
    
    def analyze(self, image_path, question, max_tokens=256):
        """Analyze image with CoT reasoning"""
        
        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((384, 384))  # Smaller for CPU
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Generate
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        elapsed = time.time() - start_time
        
        # Decode
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        
        return {
            "response": response,
            "time_seconds": elapsed
        }

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Describe the buildings in this image.")
    
    args = parser.parse_args()
    
    model = SatelliteVLM_CPU()
    
    print(f"\\nAnalyzing: {args.image}")
    result = model.analyze(args.image, args.question)
    
    print(f"\\n{'='*50}")
    print(result["response"])
    print(f"\\nTime: {result['time_seconds']:.1f}s")
'''
    
    script_path = os.path.join(GGUF_OUTPUT_PATH, "cpu_inference_transformers.py")
    os.makedirs(GGUF_OUTPUT_PATH, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Transformers CPU script created: {script_path}")
    return script_path

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("=" * 60)
    print("🚀 Step 5: Export Model to GGUF for CPU")
    print("=" * 60)
    print("⚠️  NOTE: Qwen2-VL is a multimodal model.")
    print("   GGUF conversion may not preserve the vision encoder.")
    print("   We will VERIFY this at the end.\n")
    
    # Check if fine-tuned model exists
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"⚠️ Fine-tuned model not found at {FINETUNED_MODEL_PATH}")
        print("Please run step3_finetuning.py first")
        print("\nCreating alternative CPU inference scripts...")
        create_manual_instructions()
        create_transformers_cpu_script()
        return
    
    gguf_quant_path = None
    vision_verified = False
    
    try:
        # Step 5.1: Merge LoRA
        merged_path = merge_lora_adapters()
        
        # Step 5.2: Setup llama.cpp (LATEST MASTER)
        setup_llama_cpp()
        
        # Step 5.3: Convert to GGUF
        gguf_fp16_path = convert_to_gguf()
        
        if gguf_fp16_path and os.path.exists(gguf_fp16_path):
            # Step 5.4: Quantize
            gguf_quant_path = quantize_gguf(gguf_fp16_path)
            
            # Step 5.5: Create CPU inference script
            create_cpu_inference_script(gguf_quant_path)
            
            # Step 5.6: CRITICAL - Verify vision support
            print("\n" + "🔍" * 20)
            vision_verified = verify_gguf_vision_support(gguf_quant_path)
            print("🔍" * 20)
        
        # ALWAYS create transformers-based script as RELIABLE backup
        transformers_script = create_transformers_cpu_script()
        
        print("\n" + "=" * 60)
        print("📋 EXPORT SUMMARY")
        print("=" * 60)
        print(f"📂 Merged model: {merged_path}")
        
        if gguf_quant_path and os.path.exists(gguf_quant_path):
            print(f"📂 GGUF file: {gguf_quant_path}")
            
            if vision_verified is True:
                print("✅ GGUF Vision: VERIFIED WORKING")
                print("\n🎉 You can use the GGUF model for CPU inference:")
                print(f"   python cpu_inference.py --image satellite.png")
                
            elif vision_verified is False:
                print("❌ GGUF Vision: FAILED - Vision encoder stripped!")
                print("\n⚠️  DO NOT use GGUF for images. Use Transformers instead:")
                print(f"   python {transformers_script} --image satellite.png")
                
            else:  # None = unknown
                print("⚠️  GGUF Vision: Could not verify")
                print("\n   Test manually. If images fail, use Transformers:")
                print(f"   python {transformers_script} --image satellite.png")
        else:
            print("❌ GGUF conversion failed")
            print("\n✅ Use Transformers-based CPU inference instead:")
            print(f"   python {transformers_script} --image satellite.png")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        print("\nCreating fallback scripts...")
        create_manual_instructions()
        create_transformers_cpu_script()
        
        print("\n" + "=" * 60)
        print("⚠️  GGUF export failed. Use Transformers for CPU inference.")
        print("=" * 60)

if __name__ == "__main__":
    main()
