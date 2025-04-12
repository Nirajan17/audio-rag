# Voice RAG Assistant 🎤

A sophisticated web application that combines audio transcription with RAG (Retrieval Augmented Generation) capabilities to enable intelligent question-answering about audio content. The application features both a FastAPI-based web interface and a Streamlit interface.

## Features

- 🎯 Audio file transcription using OpenAI's Whisper model
- 💬 Interactive question-answering about audio content
- 🧠 RAG implementation using LangChain and HuggingFace embeddings
- 🤖 Powered by Groq's LLM for fast and accurate responses
- 💾 Session-based conversation memory
- 🌐 Modern web interface with real-time updates
- 📱 Responsive design for various screen sizes

## Technology Stack

- **Backend Framework**: FastAPI
- **Alternative Interface**: Streamlit
- **Audio Processing**: OpenAI Whisper
- **Language Model**: Groq (llama-3.1-8b-instant)
- **Embeddings**: HuggingFace (sentence-transformers/all-mpnet-base-v2)
- **Vector Store**: LangChain's InMemoryVectorStore
- **Frontend**: HTML, CSS, JavaScript

## Prerequisites

- Python 3.10 or higher
- GROQ API key
- Sufficient disk space for model storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Voice-RAG
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your GROQ API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

### FastAPI Interface

1. Start the FastAPI server:
```bash
python voice_rag.py
```

2. Open your browser and navigate to `http://localhost:8001`
3. Enter your GROQ API key
4. Upload an audio file (supported formats: .mp3, .wav, .m4a, .ogg)
5. Wait for the processing to complete
6. Start asking questions about the audio content

### Streamlit Interface

1. Start the Streamlit app:
```bash
streamlit run voice.py
```

2. The application will open in your default web browser
3. Upload an audio file and wait for processing
4. Use the chat interface to ask questions about the audio content

## Features in Detail

### Audio Processing
- Utilizes OpenAI's Whisper model for accurate audio transcription
- Supports multiple audio formats
- Automatic handling of temporary files and cleanup

### RAG Implementation
- Text splitting with optimal chunk sizes and overlap
- Efficient vector embeddings using HuggingFace's sentence transformers
- Semantic search for relevant context retrieval

### Conversation Management
- Session-based memory for contextual conversations
- Persistent chat history within sessions
- Clean session management for new audio uploads

### User Interface
- Real-time status updates
- Error handling with user-friendly messages
- Responsive design for desktop and mobile devices
- Smooth animations and transitions

## API Endpoints

- `GET /`: Serves the web interface
- `POST /upload`: Handles audio file uploads and processing
- `POST /query`: Processes questions and returns AI-generated answers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- OpenAI Whisper for audio transcription
- Groq for the LLM API
- HuggingFace for embeddings
- LangChain for the RAG framework
- FastAPI and Streamlit teams for the web frameworks