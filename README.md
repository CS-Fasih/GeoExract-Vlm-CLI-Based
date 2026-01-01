# 🛰️ GeoExtract-VLM: Satellite Imagery Analysis using Vision Language Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**A Vision Language Model fine-tuned for geospatial analysis and satellite imagery interpretation**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Model Architecture](#model-architecture) • [Roadmap](#roadmap)

</div>

---

## 🎯 Overview

GeoExtract-VLM is a specialized Vision Language Model designed for automated analysis of satellite and aerial imagery. Built on the Qwen2-VL architecture, this model enables natural language querying of geospatial data, making satellite imagery analysis accessible through conversational AI.

### Key Capabilities

- 🏢 **Building Detection & Counting** - Identify and count structures in satellite images
- 🏘️ **Urban Density Assessment** - Analyze population density and urban development patterns
- 🛣️ **Infrastructure Analysis** - Detect roads, transportation networks, and utilities
- 🌍 **Land Use Classification** - Categorize areas as residential, commercial, industrial, etc.
- 📝 **Natural Language Interaction** - Ask questions about images in plain English

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **CLI Interface** | Interactive terminal-based image analysis |
| **GGUF Format** | Optimized for CPU inference, no GPU required |
| **Chain-of-Thought** | Structured reasoning for accurate analysis |
| **Lightweight** | ~1GB model size with 4-bit quantization |
| **Offline Ready** | Works without internet connection |

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- 4GB+ RAM recommended
- ~2GB disk space

### Quick Start

```bash
# Clone the repository
git clone https://github.com/CS-Fasih/GeoExract-Vlm-CLI-Based.git
cd GeoExract-Vlm-CLI-Based

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the CLI
python vlm_inference.py
```

### Model Download

The GGUF model file (~940MB) needs to be downloaded separately:
```bash
# Download from releases or contact maintainer
# Place in project root: qwen2vl-satellite-q4_k_m.gguf
```

---

## 💻 Usage

### Interactive CLI Mode

```bash
python vlm_inference.py
```

**Menu Options:**
```
[1] 📷 Load Image & Ask Questions
[2] 🔍 Quick Auto-Analysis
[3] 💬 Text Chat Mode
[4] 📊 Model Information
[5] 🚪 Exit
```

### Example Queries

```python
# Building Analysis
"How many buildings can you identify in this satellite image?"

# Urban Density
"Assess the urban density and development pattern in this area"

# Infrastructure
"Identify the road network and transportation infrastructure"

# Land Classification
"What type of land use is predominant in this image?"
```

### Programmatic Usage

```python
from llama_cpp import Llama

# Load model
model = Llama(
    model_path="qwen2vl-satellite-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4
)

# Analyze
prompt = """<|im_start|>system
You are a geospatial analyst specializing in satellite imagery.<|im_end|>
<|im_start|>user
Describe the urban features in this satellite image.<|im_end|>
<|im_start|>assistant
"""

response = model(prompt, max_tokens=512)
print(response['choices'][0]['text'])
```

---

## 🏗️ Model Architecture

| Component | Specification |
|-----------|--------------|
| **Base Model** | Qwen2-VL-2B-Instruct |
| **Fine-tuning Method** | QLoRA (4-bit quantization) |
| **LoRA Rank** | 64 |
| **Training Framework** | Transformers + PEFT |
| **Export Format** | GGUF (llama.cpp compatible) |
| **Quantization** | q4_k_m |
| **Model Size** | ~940MB |

### Training Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data           │────▶│  Fine-tuning    │────▶│  Export         │
│  Acquisition    │     │  with QLoRA     │     │  to GGUF        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   SpaceNet               Qwen2-VL-2B              q4_k_m
   Dataset               + LoRA Adapters          Quantized
```

---

## 📁 Project Structure

```
GeoExtract-VLM/
├── 📄 vlm_inference.py          # Interactive CLI application
├── 📄 step1_data_acquisition.py # Data download scripts
├── 📄 step2_preprocessing.py    # Dataset preparation
├── 📄 step3_finetuning.py       # Model training
├── 📄 step4_inference.py        # Inference utilities
├── 📄 step5_export_gguf.py      # GGUF export pipeline
├── 📓 VLM_Satellite_Complete_Pipeline.ipynb  # Full training notebook
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore               # Git ignore rules
└── 📄 README.md                # Documentation
```

---

## 🗺️ Roadmap

### Phase 1: CLI Application ✅
- [x] Fine-tuned VLM model
- [x] Interactive terminal interface
- [x] GGUF export for CPU inference
- [x] Basic image analysis capabilities

### Phase 2: Enhanced Model (In Progress)
- [ ] Expand training dataset (multi-region coverage)
- [ ] Improve building detection accuracy
- [ ] Add segmentation capabilities
- [ ] Multi-language support

### Phase 3: Full-Stack Web Application (Planned)
- [ ] REST API backend (FastAPI/Flask)
- [ ] React/Next.js frontend
- [ ] Map integration (Leaflet/Mapbox)
- [ ] User authentication
- [ ] Batch processing support
- [ ] Cloud deployment (AWS/GCP)

### Phase 4: Advanced Features (Future)
- [ ] Real-time satellite feed analysis
- [ ] Change detection over time
- [ ] Custom region training
- [ ] Mobile application

---

## 🔧 Technical Requirements

### Hardware (Minimum)
- **CPU**: 4+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

### Software
- Python 3.10+
- llama-cpp-python
- Pillow
- rich (for CLI UI)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Inference Time (CPU) | 20-60 seconds |
| Memory Usage | ~2GB |
| Model Load Time | ~5 seconds |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Muhammad Fasih**
- GitHub: [@CS-Fasih](https://github.com/CS-Fasih)

---

## 🙏 Acknowledgments

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Base model architecture
- [SpaceNet](https://spacenet.ai/) - Satellite imagery dataset
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and inference
- [Hugging Face](https://huggingface.co/) - Transformers library

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ for geospatial AI research

</div>
