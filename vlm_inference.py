#!/usr/bin/env python3
"""
🛰️ VLM Satellite Image Analyzer
================================
Interactive terminal-based tool to analyze satellite images
using your fine-tuned Qwen2-VL model.

Usage:
    python vlm_inference.py

Features:
    1. Load image and ask questions interactively
    2. Analyze image and get automatic description
"""

import os
import sys
import base64
from io import BytesIO
from pathlib import Path

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better UI...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint

from PIL import Image

console = Console()

# ==========================================
# Configuration
# ==========================================
MODEL_PATH = Path(__file__).parent / "qwen2vl-satellite-q4_k_m.gguf"
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_THREADS = 4

# ==========================================
# Model Loading
# ==========================================
def load_model():
    """Load the GGUF model"""
    console.print("\n[bold cyan]📦 Loading VLM Model...[/bold cyan]")
    
    if not MODEL_PATH.exists():
        console.print(f"[bold red]❌ Model not found at: {MODEL_PATH}[/bold red]")
        sys.exit(1)
    
    try:
        from llama_cpp import Llama
        
        model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=DEFAULT_CONTEXT_SIZE,
            n_threads=DEFAULT_THREADS,
            n_gpu_layers=0,  # CPU only
            verbose=False
        )
        
        console.print("[bold green]✅ Model loaded successfully![/bold green]")
        return model
    
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load model: {e}[/bold red]")
        sys.exit(1)

# ==========================================
# Image Processing
# ==========================================
def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    try:
        img = Image.open(image_path).convert('RGB')
        # Resize for faster processing
        img.thumbnail((512, 512))
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        console.print(f"[bold red]❌ Error loading image: {e}[/bold red]")
        return None

def display_image_info(image_path: str):
    """Display image information"""
    try:
        img = Image.open(image_path)
        table = Table(title="📷 Image Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("File", os.path.basename(image_path))
        table.add_row("Size", f"{img.size[0]} x {img.size[1]} pixels")
        table.add_row("Mode", img.mode)
        table.add_row("Format", img.format or "Unknown")
        
        file_size = os.path.getsize(image_path)
        table.add_row("File Size", f"{file_size/1024:.1f} KB")
        
        console.print(table)
        return True
    except Exception as e:
        console.print(f"[bold red]❌ Error reading image: {e}[/bold red]")
        return False

# ==========================================
# Inference
# ==========================================
def generate_response(model, question: str, image_b64: str = None) -> str:
    """Generate response from the model"""
    
    system_prompt = """You are a geospatial analyst AI specialized in satellite imagery analysis.
Analyze step-by-step:
1. Identify terrain and land coverage
2. Count and classify visible structures
3. Assess urban density and infrastructure
4. Note distinctive features
Then provide a detailed answer."""

    if image_b64:
        # With image
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
{question}<|im_end|>
<|im_start|>assistant
"""
    else:
        # Text only
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        response = model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    except Exception as e:
        return f"Error generating response: {e}"

# ==========================================
# Interactive Menu
# ==========================================
def show_main_menu():
    """Display main menu"""
    console.print("\n" + "="*60)
    console.print("[bold magenta]🛰️  VLM SATELLITE IMAGE ANALYZER[/bold magenta]", justify="center")
    console.print("="*60)
    
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", width=5)
    table.add_column("Description", style="white")
    
    table.add_row("[1]", "📷 Load image and ask questions")
    table.add_row("[2]", "🔍 Quick analyze (auto description)")
    table.add_row("[3]", "💬 Text-only chat (no image)")
    table.add_row("[4]", "📊 Model information")
    table.add_row("[5]", "🚪 Exit")
    
    console.print(table)
    console.print("="*60)

def interactive_image_qa(model):
    """Interactive Q&A with an image"""
    console.print("\n[bold cyan]📷 IMAGE Q&A MODE[/bold cyan]")
    
    # Get image path
    image_path = Prompt.ask("\n[yellow]Enter image path[/yellow]")
    
    if not os.path.exists(image_path):
        console.print(f"[bold red]❌ File not found: {image_path}[/bold red]")
        return
    
    # Display image info
    if not display_image_info(image_path):
        return
    
    # Encode image
    console.print("\n[cyan]🔄 Processing image...[/cyan]")
    image_b64 = encode_image_base64(image_path)
    
    if not image_b64:
        return
    
    console.print("[green]✅ Image loaded![/green]")
    console.print("\n[yellow]💡 Ask questions about the image (type 'back' to return)[/yellow]")
    
    # Q&A loop
    while True:
        question = Prompt.ask("\n[bold blue]❓ Your question[/bold blue]")
        
        if question.lower() in ['back', 'exit', 'quit', 'q']:
            break
        
        console.print("\n[cyan]🤔 Analyzing...[/cyan]")
        
        response = generate_response(model, question, image_b64)
        
        console.print(Panel(
            response,
            title="[bold green]📝 Answer[/bold green]",
            border_style="green"
        ))

def quick_analyze(model):
    """Quick automatic analysis of an image"""
    console.print("\n[bold cyan]🔍 QUICK ANALYZE MODE[/bold cyan]")
    
    # Get image path
    image_path = Prompt.ask("\n[yellow]Enter image path[/yellow]")
    
    if not os.path.exists(image_path):
        console.print(f"[bold red]❌ File not found: {image_path}[/bold red]")
        return
    
    # Display image info
    display_image_info(image_path)
    
    # Encode image
    console.print("\n[cyan]🔄 Processing image...[/cyan]")
    image_b64 = encode_image_base64(image_path)
    
    if not image_b64:
        return
    
    # Predefined analysis questions
    questions = [
        "Describe what you see in this satellite image.",
        "How many buildings can you identify?",
        "What is the urban density level (low/medium/high)?",
    ]
    
    console.print("\n[bold green]📊 ANALYSIS RESULTS[/bold green]")
    console.print("="*60)
    
    for i, q in enumerate(questions, 1):
        console.print(f"\n[cyan]🔍 Analyzing ({i}/{len(questions)}): {q}[/cyan]")
        response = generate_response(model, q, image_b64)
        
        console.print(Panel(
            response,
            title=f"[bold yellow]Q{i}: {q[:50]}...[/bold yellow]",
            border_style="yellow"
        ))

def text_chat(model):
    """Text-only chat mode"""
    console.print("\n[bold cyan]💬 TEXT CHAT MODE[/bold cyan]")
    console.print("[yellow]Ask any question (type 'back' to return)[/yellow]")
    
    while True:
        question = Prompt.ask("\n[bold blue]❓ Your question[/bold blue]")
        
        if question.lower() in ['back', 'exit', 'quit', 'q']:
            break
        
        console.print("\n[cyan]🤔 Thinking...[/cyan]")
        
        response = generate_response(model, question)
        
        console.print(Panel(
            response,
            title="[bold green]📝 Answer[/bold green]",
            border_style="green"
        ))

def show_model_info():
    """Display model information"""
    console.print("\n[bold cyan]📊 MODEL INFORMATION[/bold cyan]")
    
    table = Table(title="🛰️ VLM Satellite Model")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model Name", "qwen2vl-satellite-q4_k_m")
    table.add_row("Base Model", "Qwen/Qwen2-VL-2B-Instruct")
    table.add_row("Format", "GGUF v3")
    table.add_row("Quantization", "q4_k_m (4-bit)")
    table.add_row("Tensor Count", "338")
    
    if MODEL_PATH.exists():
        size = MODEL_PATH.stat().st_size
        table.add_row("File Size", f"{size/1024**3:.2f} GB")
        table.add_row("Status", "✅ Available")
    else:
        table.add_row("Status", "❌ Not Found")
    
    table.add_row("Training Data", "SpaceNet 7 Rotterdam")
    table.add_row("Capabilities", "Building counting, Density analysis, Infrastructure detection")
    
    console.print(table)

# ==========================================
# Main
# ==========================================
def main():
    console.print(Panel.fit(
        "[bold cyan]🛰️ VLM SATELLITE IMAGE ANALYZER[/bold cyan]\n"
        "[dim]Fine-tuned Qwen2-VL for satellite imagery analysis[/dim]",
        border_style="cyan"
    ))
    
    # Load model
    model = load_model()
    
    # Main loop
    while True:
        show_main_menu()
        
        choice = Prompt.ask(
            "[bold yellow]Select option[/bold yellow]",
            choices=["1", "2", "3", "4", "5"],
            default="1"
        )
        
        if choice == "1":
            interactive_image_qa(model)
        elif choice == "2":
            quick_analyze(model)
        elif choice == "3":
            text_chat(model)
        elif choice == "4":
            show_model_info()
        elif choice == "5":
            console.print("\n[bold green]👋 Goodbye![/bold green]")
            break

if __name__ == "__main__":
    main()
