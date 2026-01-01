# 🛰️ GeoExtract-VLM: Satellite Imagery Analysis Web Application

<div align="center">

![React](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react)
![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg?logo=node.js)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python)
![MongoDB](https://img.shields.io/badge/MongoDB-6+-47A248.svg?logo=mongodb)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A ChatGPT-like web application for satellite imagery analysis using Vision Language Models**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Roadmap](#roadmap)

</div>

---

## 🎯 Overview

GeoExtract-VLM is a full-stack web application that brings the power of Vision Language Models to satellite imagery analysis. Built with the MERN stack (MongoDB, Express, React, Node.js) and powered by a fine-tuned Qwen2-VL model, it provides a ChatGPT-like conversational interface for geospatial analysis.

### Key Capabilities

- 🏢 **Building Detection & Counting** - Identify and count structures in satellite images
- 🏘️ **Urban Density Assessment** - Analyze population density and urban development patterns
- 🛣️ **Infrastructure Analysis** - Detect roads, transportation networks, and utilities
- 🌍 **Land Use Classification** - Categorize areas as residential, commercial, industrial, etc.
- 💬 **Conversational AI Interface** - ChatGPT-like experience for image analysis

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **ChatGPT-like UI** | Modern, responsive chat interface |
| **Image Upload** | Drag & drop satellite imagery |
| **Chat History** | MongoDB-backed conversation storage |
| **Dark/Light Mode** | Toggle between themes |
| **Real-time Analysis** | Instant VLM-powered responses |
| **GGUF Model** | Optimized for CPU inference |

<div align="center">
<img src="docs/screenshot.png" alt="GeoExtract-VLM Screenshot" width="800"/>
</div>

---

## 🚀 Installation

### Prerequisites

- Node.js 18+
- Python 3.10+
- MongoDB (optional, for chat history)
- 4GB+ RAM recommended

### Quick Start

```bash
# Clone the repository
git clone https://github.com/CS-Fasih/GeoExract-Vlm-CLI-Based.git
cd GeoExract-Vlm-CLI-Based
```

#### 1️⃣ Backend Setup

```bash
cd backend

# Install Node.js dependencies
npm install

# Install Python dependencies (for model service)
pip install -r requirements.txt

# Start the Express server
npm run dev

# In a new terminal, start the model service
python model_service.py
```

#### 2️⃣ Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### 3️⃣ Access the Application

Open your browser and navigate to: **http://localhost:5173**

---

## 💻 Usage

### Web Interface

1. **Upload Image**: Click the 📷 button to upload a satellite image
2. **Ask Questions**: Type your query in the chat input
3. **Get Analysis**: Receive detailed VLM-powered analysis
4. **View History**: Access previous conversations in the sidebar

### Example Queries

```
🏢 "How many buildings can you identify in this image?"
🏘️ "Analyze the urban density and development pattern"
🛣️ "Identify the road network and infrastructure"
🌍 "What type of land use is predominant?"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                    http://localhost:5173                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────────┐
│                    Backend (Express.js)                          │
│                    http://localhost:5000                         │
│  • File uploads (Multer)                                         │
│  • Chat API endpoints                                            │
│  • MongoDB integration                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────┐
│                  Model Service (FastAPI)                         │
│                    http://localhost:8000                         │
│  • GGUF model loading                                            │
│  • Image + text inference                                        │
│  • llama-cpp-python                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, Vite, Axios, Lucide Icons |
| **Backend** | Node.js, Express.js, Multer, Mongoose |
| **Model Service** | Python, FastAPI, llama-cpp-python |
| **Database** | MongoDB |
| **AI Model** | Qwen2-VL-2B (GGUF q4_k_m) |

---

## 📁 Project Structure

```
GeoExtract-VLM/
├── 📂 frontend/                 # React application
│   ├── src/
│   │   ├── App.jsx             # Main chat component
│   │   └── App.css             # ChatGPT-like styles
│   └── package.json
│
├── 📂 backend/                  # Express + Python services
│   ├── server.js               # Express API server
│   ├── model_service.py        # FastAPI model service
│   ├── requirements.txt        # Python dependencies
│   └── package.json            # Node dependencies
│
├── 📂 training/                 # Model training scripts
│   ├── step1_data_acquisition.py
│   ├── step2_preprocessing.py
│   ├── step3_finetuning.py
│   ├── step4_inference.py
│   └── step5_export_gguf.py
│
├── 📓 VLM_Satellite_Complete_Pipeline.ipynb
├── 🤖 qwen2vl-satellite-q4_k_m.gguf  # Fine-tuned model
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🗺️ Roadmap

### Phase 1: Core Web Application ✅
- [x] Fine-tuned VLM model (Qwen2-VL-2B)
- [x] React frontend with ChatGPT-like UI
- [x] Express.js backend with file uploads
- [x] FastAPI model service
- [x] Dark/Light mode support
- [x] Basic chat functionality

### Phase 2: Enhanced Features (In Progress)
- [ ] User authentication (JWT)
- [ ] Chat history persistence
- [ ] Image gallery/history
- [ ] Export analysis reports
- [ ] Batch image processing

### Phase 3: Advanced Integration (Planned)
- [ ] Map integration (Leaflet/Mapbox)
- [ ] Drawing tools for ROI selection
- [ ] GeoJSON/KML export
- [ ] Multi-image comparison
- [ ] Time-series analysis

### Phase 4: Production Ready (Future)
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Cloud hosting (AWS/GCP/Azure)
- [ ] API rate limiting
- [ ] Admin dashboard
- [ ] Mobile responsive optimization

---

## 🔧 Configuration

### Environment Variables

**Backend (.env)**
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/geoextract
MODEL_SERVICE_URL=http://localhost:8000
```

### Model Configuration

The GGUF model is loaded with these settings:
```python
Llama(
    model_path="qwen2vl-satellite-q4_k_m.gguf",
    n_ctx=2048,      # Context window
    n_threads=4,     # CPU threads
    n_gpu_layers=0   # CPU-only inference
)
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Frontend Load Time | < 1 second |
| API Response Time | < 100ms (without model) |
| Model Inference | 20-60 seconds (CPU) |
| Memory Usage | ~2GB |

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
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and inference
- [OpenAI ChatGPT](https://chat.openai.com) - UI inspiration
- [Hugging Face](https://huggingface.co/) - Transformers library

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ for geospatial AI research

</div>
