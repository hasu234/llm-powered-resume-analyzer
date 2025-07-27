# 🤖 LLM-Powered Resume Scanner

An intelligent AI-powered recruitment tool that revolutionizes how hiring managers interact with resume data using advanced Retrieval-Augmented Generation (RAG) technology.

## 📌 Overview

Traditional recruitment processes generate vast amounts of unstructured data—CVs, job descriptions, and candidate information. Our AI-powered RAG system transforms this challenge into an opportunity, allowing hiring managers to:

- **Search candidates** based on job descriptions intelligently
- **Interact with specific CVs** using natural language queries
- **Process multiple resumes** simultaneously with vector database technology
- **Integrate seamlessly** with Google Drive for team collaboration
- **Get instant insights** without endless document searching

## ✨ Key Features

### 🧠 Advanced RAG Modes
- **Generic RAG**: Standard retrieval and generation for general queries
- **RAG Fusion**: Enhanced multi-query approach for deeper insights

### 📁 Flexible File Sources
- **Local Files**: Upload and process PDFs directly
- **Google Drive Integration**: Team-based resume management with automatic sync

### 🔍 Intelligent Search Capabilities
- Natural language queries for candidate matching
- Semantic search across resume content
- Context-aware responses with source attribution
- Real-time conversation management

### 👥 Team Collaboration
- Multi-team workspace support
- Centralized resume database per team
- Automatic vector database synchronization
- Secure Google Drive integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12
- CUDA-compatible GPU (optional, for better performance)
- OpenAI API key
- Google Drive API credentials (for Drive integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-powered-resume-scanner
   ```

2. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt

   # Or using uv (recommended)
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LLM_MODEL=gpt-4o-mini  # or your preferred model
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

4. **Configure Google Drive (Optional)**
   - Place your `credentials.json` file in the `utils/` directory
   - Configure team folders in `config.ini`

5. **Run the application**
   ```bash
   # Using the start script
   ./scripts/start.sh

   # Or directly with Streamlit
   streamlit run interface.py
   ```

## 📖 Usage Guide

### 1. **Select RAG Mode**
Choose between:
- **Generic RAG**: Standard retrieval for straightforward queries
- **RAG Fusion**: Advanced multi-perspective analysis

### 2. **Choose File Source**
- **Local Files**: Upload PDFs directly for immediate processing
- **Google Drive**: Access team-organized resume databases

### 3. **Start Chatting**
Begin asking questions like:
- "Find candidates with Python and machine learning experience"
- "Show me frontend developers with React skills"
- "Who has experience in fintech companies?"

### 4. **Manage Teams** (Google Drive mode)
- Create new teams for different departments
- Upload resumes to specific team folders
- Sync and maintain vector databases automatically

## 🛠️ Technical Architecture

### Core Components

- **RAG Pipeline**: Advanced retrieval system using FAISS vector database
- **LLM Integration**: OpenAI GPT models for intelligent responses
- **Document Processing**: PDF parsing and text extraction
- **Embedding Model**: Sentence transformers for semantic search
- **Vector Database**: FAISS for efficient similarity search

### File Structure

```
llm-powered-resume-scanner/
├── interface.py              # Main Streamlit interface
├── fastapi-app.py           # FastAPI application (alternative interface)
├── config.ini               # Configuration settings
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
├── scripts/                # Utility scripts
│   ├── install.sh          # Installation script
│   ├── start.sh            # Application starter
│   └── check_cuda.py       # CUDA availability checker
└── utils/                  # Core utilities
    ├── llm_agent.py        # LLM chatbot implementation
    ├── retriever.py        # RAG retrieval system
    ├── convert_pdf.py      # PDF processing
    ├── manage_driver.py    # Google Drive management
    ├── configuration.py    # Config file handling
    └── chatbot_verbosity.py # Response formatting
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `LLM_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |

### Google Drive Setup

1. Create a Google Cloud project
2. Enable Google Drive API
3. Create service account credentials
4. Download `credentials.json` to `utils/` directory
5. Configure team folders in `config.ini`

## 💡 Example Queries

- **Skill-based search**: "Find developers with TypeScript and Node.js experience"
- **Experience level**: "Show me senior candidates with 5+ years of experience"
- **Industry focus**: "Who has worked in healthcare or medical technology?"
- **Education filter**: "Find candidates with computer science degrees"
- **Location-based**: "Show me candidates from New York or remote workers"

## 🧪 Development

### Running Tests

```bash
# Run CUDA availability check
python scripts/check_cuda.py

# Test the application
streamlit run interface.py
```

### Adding New Features

1. **Custom Retrieval Logic**: Modify `utils/retriever.py`
2. **LLM Behavior**: Update `utils/llm_agent.py`
3. **UI Components**: Edit `interface.py`
4. **File Processing**: Enhance `utils/convert_pdf.py`

## 📊 Performance Optimization

- **GPU Acceleration**: CUDA support for faster embedding generation
- **Vector Database**: FAISS for efficient similarity search
- **Chunking Strategy**: Optimized text splitting for better retrieval
- **Caching**: Session state management for improved user experience

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📄 License

This project is developed by [Silicon Orchard Ltd](https://www.siliconorchard.com).

## 🆘 Support

For issues and questions:
- Check the troubleshooting section in this README
- Review the code documentation
- Contact the development team

## 🚀 Future Roadmap

- [ ] Multi-language resume support
- [ ] Advanced analytics dashboard
- [ ] Integration with popular ATS systems
- [ ] Batch processing capabilities
- [ ] Enhanced security features
- [ ] Mobile-responsive interface

---

**Transform your recruitment process with AI-powered intelligence. Start finding the perfect candidates today!** 🎯