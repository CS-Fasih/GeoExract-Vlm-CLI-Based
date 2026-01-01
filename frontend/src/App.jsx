import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { 
  Send, 
  ImagePlus, 
  Trash2, 
  Plus, 
  MessageSquare, 
  Satellite, 
  X,
  Loader2,
  Menu,
  Moon,
  Sun
} from 'lucide-react'
import './App.css'

const API_URL = 'http://localhost:5000/api'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [chats, setChats] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [darkMode, setDarkMode] = useState(true)
  
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Load chats on mount
  useEffect(() => {
    loadChats()
  }, [])

  // Apply dark mode
  useEffect(() => {
    document.body.classList.toggle('light-mode', !darkMode)
  }, [darkMode])

  const loadChats = async () => {
    try {
      const response = await axios.get(`${API_URL}/chats`)
      setChats(response.data)
    } catch (error) {
      console.log('Could not load chats')
    }
  }

  const createNewChat = async () => {
    setMessages([])
    setCurrentSessionId(null)
    setImage(null)
    setImagePreview(null)
    setInput('')
  }

  const loadChat = async (sessionId) => {
    try {
      const response = await axios.get(`${API_URL}/chats/${sessionId}`)
      setMessages(response.data.messages || [])
      setCurrentSessionId(sessionId)
    } catch (error) {
      console.error('Error loading chat:', error)
    }
  }

  const deleteChat = async (sessionId, e) => {
    e.stopPropagation()
    try {
      await axios.delete(`${API_URL}/chats/${sessionId}`)
      setChats(chats.filter(chat => chat.sessionId !== sessionId))
      if (currentSessionId === sessionId) {
        createNewChat()
      }
    } catch (error) {
      console.error('Error deleting chat:', error)
    }
  }

  const handleImageSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const removeImage = () => {
    setImage(null)
    setImagePreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() && !image) return

    const userMessage = {
      role: 'user',
      content: input,
      image: imagePreview
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    const formData = new FormData()
    formData.append('message', input || 'Analyze this satellite image')
    if (image) {
      formData.append('image', image)
    }
    if (currentSessionId) {
      formData.append('sessionId', currentSessionId)
    }

    // Clear image after sending
    setImage(null)
    setImagePreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }

    try {
      const response = await axios.post(`${API_URL}/chat`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response
      }

      setMessages(prev => [...prev, assistantMessage])
      loadChats() // Refresh chat list
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Error: Could not connect to the server. Make sure the backend is running on port 5000.'
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const formatMessage = (content) => {
    // Simple markdown-like formatting
    return content
      .split('\n')
      .map((line, i) => {
        // Headers
        if (line.startsWith('### ')) {
          return <h4 key={i} className="message-h4">{line.slice(4)}</h4>
        }
        if (line.startsWith('## ')) {
          return <h3 key={i} className="message-h3">{line.slice(3)}</h3>
        }
        if (line.startsWith('# ')) {
          return <h2 key={i} className="message-h2">{line.slice(2)}</h2>
        }
        // Bold
        if (line.includes('**')) {
          const parts = line.split(/\*\*(.*?)\*\*/g)
          return (
            <p key={i}>
              {parts.map((part, j) => 
                j % 2 === 1 ? <strong key={j}>{part}</strong> : part
              )}
            </p>
          )
        }
        // List items
        if (line.startsWith('- ')) {
          return <li key={i}>{line.slice(2)}</li>
        }
        if (line.match(/^\d+\. /)) {
          return <li key={i}>{line.slice(line.indexOf(' ') + 1)}</li>
        }
        // Empty line
        if (!line.trim()) {
          return <br key={i} />
        }
        return <p key={i}>{line}</p>
      })
  }

  return (
    <div className={`app ${darkMode ? 'dark' : 'light'}`}>
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={createNewChat}>
            <Plus size={20} />
            <span>New Chat</span>
          </button>
        </div>
        
        <div className="chat-list">
          {chats.map((chat) => (
            <div 
              key={chat.sessionId}
              className={`chat-item ${currentSessionId === chat.sessionId ? 'active' : ''}`}
              onClick={() => loadChat(chat.sessionId)}
            >
              <MessageSquare size={16} />
              <span className="chat-title">{chat.title || 'New Chat'}</span>
              <button 
                className="delete-chat-btn"
                onClick={(e) => deleteChat(chat.sessionId, e)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <button 
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Header */}
        <header className="header">
          <button 
            className="menu-btn"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu size={24} />
          </button>
          <div className="header-title">
            <Satellite size={28} />
            <h1>GeoExtract-VLM</h1>
          </div>
          <div className="header-subtitle">Satellite Imagery Analysis</div>
        </header>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <Satellite size={64} className="welcome-icon" />
              <h2>GeoExtract-VLM</h2>
              <p>Upload satellite imagery and ask questions about buildings, infrastructure, land use, and more.</p>
              
              <div className="example-prompts">
                <h3>Try asking:</h3>
                <div className="prompt-grid">
                  <button 
                    className="prompt-card"
                    onClick={() => setInput("How many buildings can you identify in this image?")}
                  >
                    🏢 Count buildings in the image
                  </button>
                  <button 
                    className="prompt-card"
                    onClick={() => setInput("Analyze the urban density and development pattern")}
                  >
                    🏘️ Assess urban density
                  </button>
                  <button 
                    className="prompt-card"
                    onClick={() => setInput("Identify the road network and infrastructure")}
                  >
                    🛣️ Analyze infrastructure
                  </button>
                  <button 
                    className="prompt-card"
                    onClick={() => setInput("What type of land use is predominant?")}
                  >
                    🌍 Classify land use
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : '🛰️'}
                  </div>
                  <div className="message-content">
                    {msg.image && (
                      <div className="message-image">
                        <img src={msg.image} alt="Uploaded" />
                      </div>
                    )}
                    <div className="message-text">
                      {formatMessage(msg.content)}
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="message assistant">
                  <div className="message-avatar">🛰️</div>
                  <div className="message-content">
                    <div className="loading-indicator">
                      <Loader2 className="spin" size={20} />
                      <span>Analyzing...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="input-container">
          {imagePreview && (
            <div className="image-preview">
              <img src={imagePreview} alt="Preview" />
              <button className="remove-image-btn" onClick={removeImage}>
                <X size={16} />
              </button>
            </div>
          )}
          
          <form onSubmit={sendMessage} className="input-form">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageSelect}
              accept="image/*,.tif,.tiff"
              style={{ display: 'none' }}
            />
            
            <button 
              type="button"
              className="attach-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              <ImagePlus size={22} />
            </button>
            
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about satellite imagery..."
              className="message-input"
              disabled={isLoading}
            />
            
            <button 
              type="submit"
              className="send-btn"
              disabled={isLoading || (!input.trim() && !image)}
            >
              {isLoading ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
            </button>
          </form>
          
          <p className="disclaimer">
            GeoExtract-VLM - Satellite Imagery Analysis powered by Vision Language Models
          </p>
        </div>
      </main>
    </div>
  )
}

export default App
