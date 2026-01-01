"""
GeoExtract-VLM Model Service
FastAPI service for VLM inference
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import base64
import os
import sys

# Add parent directory to path for model access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="GeoExtract-VLM Model Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qwen2vl-satellite-q4_k_m.gguf")


def load_model():
    """Load the GGUF model"""
    global model
    if model is not None:
        return model
    
    try:
        from llama_cpp import Llama
        print(f"🔄 Loading model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # CPU only
            verbose=False
        )
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def analyze_with_model(prompt: str, image: Image.Image = None) -> str:
    """Run inference with the VLM model"""
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            return "❌ Model failed to load. Please check if the GGUF file exists."
    
    # Build the prompt
    system_prompt = """You are GeoExtract-VLM, an expert geospatial analyst specializing in satellite and aerial imagery analysis. 
Analyze images to identify:
- Buildings, structures, and their count
- Urban density and development patterns
- Roads, infrastructure, and transportation networks
- Land use types (residential, commercial, industrial, agricultural)
- Natural features (water bodies, vegetation, terrain)

Provide detailed, structured analysis with clear observations."""

    if image:
        # For image analysis
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
[Image attached: satellite/aerial imagery]

{prompt}<|im_end|>
<|im_start|>assistant
"""
    else:
        # Text-only query
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        response = model(
            full_prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"❌ Inference error: {str(e)}"


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n🛰️  GeoExtract-VLM Model Service Starting...")
    print("=" * 50)
    load_model()


@app.get("/")
async def root():
    return {"status": "ok", "service": "GeoExtract-VLM Model Service"}


@app.get("/health")
async def health():
    global model
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/analyze")
async def analyze(
    message: str = Form(...),
    image: UploadFile = File(None)
):
    """Analyze satellite image with optional prompt"""
    try:
        pil_image = None
        
        if image:
            # Read and process image
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        
        # Run analysis
        response = analyze_with_model(message, pil_image)
        
        return JSONResponse({
            "success": True,
            "response": response,
            "has_image": image is not None
        })
        
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@app.post("/chat")
async def chat(message: str = Form(...)):
    """Text-only chat endpoint"""
    try:
        response = analyze_with_model(message)
        return JSONResponse({
            "success": True,
            "response": response
        })
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     🛰️  GeoExtract-VLM Model Service                  ║
    ║     Satellite Imagery Analysis with Vision LLM        ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
