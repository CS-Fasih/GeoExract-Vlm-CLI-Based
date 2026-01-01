const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Create uploads directory
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Multer configuration for image uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|tif|tiff|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype) || file.mimetype === 'image/tiff';
    if (extname && mimetype) {
      return cb(null, true);
    }
    cb(new Error('Only image files are allowed'));
  }
});

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/geoextract')
  .then(() => console.log('✅ Connected to MongoDB'))
  .catch(err => console.log('⚠️ MongoDB connection error:', err.message, '- Running without database'));

// Chat Schema
const chatSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, index: true },
  title: { type: String, default: 'New Chat' },
  messages: [{
    role: { type: String, enum: ['user', 'assistant'], required: true },
    content: { type: String, required: true },
    image: { type: String },
    timestamp: { type: Date, default: Date.now }
  }],
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

const Chat = mongoose.model('Chat', chatSchema);

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'GeoExtract-VLM API is running' });
});

// Get all chat sessions
app.get('/api/chats', async (req, res) => {
  try {
    const chats = await Chat.find({}, 'sessionId title createdAt updatedAt')
      .sort({ updatedAt: -1 })
      .limit(50);
    res.json(chats);
  } catch (error) {
    res.json([]); // Return empty if DB not connected
  }
});

// Get single chat session
app.get('/api/chats/:sessionId', async (req, res) => {
  try {
    const chat = await Chat.findOne({ sessionId: req.params.sessionId });
    if (!chat) {
      return res.status(404).json({ error: 'Chat not found' });
    }
    res.json(chat);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Create new chat session
app.post('/api/chats', async (req, res) => {
  try {
    const sessionId = uuidv4();
    const chat = new Chat({
      sessionId,
      title: req.body.title || 'New Chat',
      messages: []
    });
    await chat.save();
    res.json(chat);
  } catch (error) {
    // If DB not connected, return a temporary session
    res.json({
      sessionId: uuidv4(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date()
    });
  }
});

// Delete chat session
app.delete('/api/chats/:sessionId', async (req, res) => {
  try {
    await Chat.deleteOne({ sessionId: req.params.sessionId });
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Upload image
app.post('/api/upload', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }
  res.json({
    success: true,
    filename: req.file.filename,
    path: `/uploads/${req.file.filename}`,
    url: `http://localhost:${PORT}/uploads/${req.file.filename}`
  });
});

// Chat with VLM model
app.post('/api/chat', upload.single('image'), async (req, res) => {
  try {
    const { message, sessionId } = req.body;
    const imagePath = req.file ? path.join(uploadsDir, req.file.filename) : null;
    const imageUrl = req.file ? `/uploads/${req.file.filename}` : null;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Call Python model service
    const modelServiceUrl = process.env.MODEL_SERVICE_URL || 'http://localhost:8000';
    
    const formData = new FormData();
    formData.append('message', message);
    
    if (imagePath && fs.existsSync(imagePath)) {
      const imageBuffer = fs.readFileSync(imagePath);
      const blob = new Blob([imageBuffer], { type: 'image/png' });
      formData.append('image', blob, req.file.filename);
    }

    let assistantResponse;
    
    try {
      const response = await fetch(`${modelServiceUrl}/analyze`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Model service error');
      }
      
      const data = await response.json();
      assistantResponse = data.response;
    } catch (modelError) {
      // Fallback response if model service is not running
      assistantResponse = generateFallbackResponse(message, imagePath !== null);
    }

    // Save to database if connected
    if (sessionId) {
      try {
        await Chat.findOneAndUpdate(
          { sessionId },
          {
            $push: {
              messages: [
                { role: 'user', content: message, image: imageUrl },
                { role: 'assistant', content: assistantResponse }
              ]
            },
            $set: { updatedAt: new Date() }
          },
          { upsert: true }
        );
      } catch (dbError) {
        console.log('DB save skipped:', dbError.message);
      }
    }

    res.json({
      success: true,
      response: assistantResponse,
      image: imageUrl
    });

  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Fallback response generator when model service is not available
function generateFallbackResponse(message, hasImage) {
  if (hasImage) {
    return `🛰️ **Image Analysis Request Received**

I've received your satellite image and query: "${message}"

⚠️ **Model Service Status**: The VLM model service is currently starting up or not connected.

**To enable full analysis:**
1. Start the model service: \`python model_service.py\`
2. Wait for the model to load (~30 seconds)
3. Try your query again

Once connected, I can provide:
- Building detection and counting
- Urban density assessment
- Infrastructure analysis
- Land use classification`;
  }
  
  return `Hello! I'm GeoExtract-VLM, a Vision Language Model specialized in satellite imagery analysis.

**How to use me:**
1. 📷 Upload a satellite or aerial image
2. 💬 Ask questions about what you see
3. 🔍 I'll analyze and provide insights

**Example queries:**
- "How many buildings are in this image?"
- "Describe the urban development pattern"
- "What type of infrastructure can you identify?"
- "Assess the population density of this area"

Upload an image to get started! 🚀`;
}

// Start server
app.listen(PORT, () => {
  console.log(`
🛰️  GeoExtract-VLM Backend Server
================================
✅ Server running on http://localhost:${PORT}
📁 Uploads directory: ${uploadsDir}
🔗 API endpoints:
   - GET  /api/health
   - GET  /api/chats
   - POST /api/chats
   - POST /api/chat
   - POST /api/upload
  `);
});
