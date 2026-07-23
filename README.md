# Chatbot: RAG-Powered AI Assistant

A production-ready chatbot built with **Streamlit**, **LangChain**, and **FAISS** that intelligently answers questions from your documents with intelligent fallback to OpenRouter.

## 🎯 What It Does

This chatbot is a **Retrieval-Augmented Generation (RAG)** system that:

- **Uploads & Indexes Documents**: Support for PDFs, images (with OCR), and text files
- **Vector Search**: Uses FAISS for fast semantic similarity search
- **Context-Aware Conversations**: Maintains chat history and formulates follow-up questions intelligently
- **Hybrid LLM Support**: 
  - Primary: HuggingFace's `google/flan-t5-base` for fast, local inference
  - Fallback: OpenRouter API for questions outside your knowledge base
- **Image OCR**: Extracts text from images using Tesseract OCR
- **Streaming Responses**: Real-time response generation for better UX

## 🏗️ Architecture

```
User Input
    ↓
[Streamlit UI] ← Chat Interface & File Upload
    ↓
[LangChain] ← Orchestration Layer
    ├─→ History-Aware Retriever (reformulates questions with context)
    ├─→ Vector Store Retrieval (FAISS)
    └─→ Document Chain (Stuff Documents pattern)
    ↓
[Local Model] (google/flan-t5-base)
    ↓
Is "I do not have that information" in response?
    ├─ YES → Fallback to OpenRouter API (mistralai/mistral-7b-instruct)
    └─ NO → Return context-based answer
    ↓
[Embeddings] (HuggingFace sentence-transformers/all-MiniLM-L6-v2)
    ↓
[FAISS Vector Database] ← Semantic search on uploaded docs
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Streamlit** | Web UI with chat interface & file uploads |
| **LangChain** | LLM orchestration & RAG pipeline |
| **FAISS** | Vector database for semantic search |
| **HuggingFace** | Embeddings + local LLM inference |
| **OpenRouter** | Fallback API for out-of-scope questions |
| **Tesseract OCR** | Extract text from images |

## 📋 Setup Steps

### Prerequisites

- Python 3.10+
- Tesseract OCR (optional, for image processing)
- OpenRouter API key (optional, for fallback model)

### 1. Clone the Repository

```bash
git clone https://github.com/SwayamGPT/Chatbot.git
cd Chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (Optional, for Image Processing)

**On macOS:**
```bash
brew install tesseract
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**On Windows:**
- Download the installer: https://github.com/UB-Mannheim/tesseract/wiki
- Install to the default location or update the path in `app.py` (line 29)

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your API key from [OpenRouter](https://openrouter.ai).

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🚀 Usage

### Basic Chat
1. Type your question in the input box
2. The chatbot searches your knowledge base first
3. If no answer is found, it queries OpenRouter for a fallback response

### Upload Documents
1. Use the **Upload & Process** sidebar
2. Select PDF, images, or text files
3. Click **Process Documents**
4. Chat about the new content

### Clear History
Click **Clear Chat History** to reset the conversation and start fresh.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | API key for OpenRouter fallback | No (fallback disabled if missing) |

### Model Settings (in `app.py`)

- **Local Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` (25MB)
- **Local LLM**: `google/flan-t5-base` (220MB)
- **Fallback Model**: `mistralai/mistral-7b-instruct:free` (via OpenRouter)
- **Vector Store**: FAISS with cosine similarity
- **Chunk Size**: 1000 characters with 200-character overlap
- **Retrieval K**: Top 3 most relevant documents

To customize, edit lines 25, 34-38, and 137 in `app.py`.

## 📦 Requirements

See `requirements.txt` for the complete list. Key dependencies:

- `streamlit>=1.28.0`
- `langchain>=0.1.0`
- `langchain-community>=0.0.10`
- `langchain-core>=0.1.0`
- `langchain-text-splitters>=0.0.1`
- `langchain-huggingface>=0.0.1`
- `faiss-cpu>=1.7.4`
- `openai>=1.3.0` (for OpenRouter)
- `python-dotenv>=1.0.0`
- `pytesseract>=0.3.10` (for OCR)
- `Pillow>=9.0.0`

## 🎨 Features

✅ **Multi-Format Support**: PDFs, images, text files  
✅ **OCR Integration**: Automatic text extraction from images  
✅ **Chat History**: Maintains conversation context  
✅ **Streaming Responses**: Real-time answer generation  
✅ **Intelligent Fallback**: Graceful degradation when knowledge base insufficient  
✅ **Local Inference**: Privacy-first option (HuggingFace models)  
✅ **Semantic Search**: Vector-based document retrieval  
✅ **Error Handling**: Graceful degradation for missing dependencies  

## 🐛 Troubleshooting

### Tesseract Not Found
If you get a "Tesseract not found" error:
- **macOS**: Run `brew install tesseract`
- **Linux**: Run `sudo apt-get install tesseract-ocr`
- **Windows**: Uncomment and update line 29 in `app.py` with your Tesseract installation path

### FAISS Installation Issues
If FAISS fails to install:
```bash
pip install faiss-cpu  # For CPU-only systems
# OR
pip install faiss-gpu  # For NVIDIA GPU systems (requires CUDA)
```

### OpenRouter API Key Not Working
- Verify your API key in `.env`
- Check your OpenRouter account has available credits
- The fallback is optional; the chatbot works without it

### Memory Issues
If running on limited hardware:
- Use `faiss-cpu` instead of GPU version
- Reduce `chunk_size` in line 128 (currently 1000)
- Reduce `k` in line 137 (currently 3) for fewer retrieved documents

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📞 Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/SwayamGPT/Chatbot/issues).

---

**Built with ❤️ using Streamlit, LangChain, and FAISS**
