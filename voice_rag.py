# app.py
import os
import re
import shutil
import time
import tempfile
from typing import Optional
from collections import defaultdict
import uuid

try:
    import whisper
except ImportError:
    try:
        import openai.whisper as whisper
    except ImportError:
        raise ImportError("Could not import whisper. Please install it using 'pip install openai-whisper'")

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from starlette.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(title="Audio RAG API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("audio-files", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store models and data
whisper_model = None
embeddings = None
vector_store = None
transcript = ""
api_key = ""
# Dictionary to store session memories
session_memories = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
))

# LLM template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

class QueryModel(BaseModel):
    question: str
    api_key: str
    session_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global whisper_model, embeddings
    
    # Load Whisper model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("medium.en")
    
    # Load embeddings
    print("Loading embeddings model...")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    
    # Initialize vector store
    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    
    print("Models loaded and ready!")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), api_key_param: str = Form(...)):
    global transcript, vector_store
    
    if not file:
        return JSONResponse(
            status_code=400,
            content={"message": "No file provided"}
        )
    
    # Save API key for later use
    global api_key
    api_key = api_key_param  # Rename the parameter in the function definition
        
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    
    try:
        # Write the file content
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        # Transcribe audio
        print("Transcribing audio...")
        result = whisper_model.transcribe(temp_file.name)
        transcript = result["text"]
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunked_texts = text_splitter.split_text(transcript)
        
        # Index documents
        print("Creating vector embeddings...")
        vector_store.add_texts(chunked_texts)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File processed successfully",
                "transcript_length": len(transcript),
                "chunks_processed": len(chunked_texts)
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )
    finally:
        # Clean up
        os.unlink(temp_file.name)

@app.post("/query")
async def query(query_data: QueryModel):
    if not vector_store:
        return JSONResponse(
            status_code=400,
            content={"message": "Please upload an audio file first"}
        )
    
    try:
        # Get or create session memory
        session_id = query_data.session_id or str(uuid.uuid4())
        memory = session_memories[session_id]
        
        # Use provided API key
        model = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=query_data.api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Retrieve relevant documents
        related_docs = vector_store.similarity_search(query_data.question)
        context = "\n\n".join([doc.page_content for doc in related_docs])
        
        # Get chat history
        chat_history = memory.load_memory_variables({})
        history_str = ""
        if "chat_history" in chat_history:
            history = chat_history["chat_history"]
            history_str = "\n".join([f"Human: {isinstance(msg, HumanMessage) and msg.content or ''}\nAssistant: {isinstance(msg, AIMessage) and msg.content or ''}" for msg in history])
        
        # Updated template with chat history
        prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context and chat history to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            
            Previous conversation:
            {chat_history}
            
            Context: {context}
            Question: {question}
            
            Answer:
        """)
        
        # Generate answer
        chain = prompt | model
        response = chain.invoke({
            "question": query_data.question,
            "context": context,
            "chat_history": history_str
        })
        
        # Save the interaction to memory
        memory.save_context(
            {"input": query_data.question},
            {"output": response.content}
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "answer": response.content,
                "session_id": session_id
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing query: {str(e)}"}
        )

@app.get("/", response_class=HTMLResponse)
async def get_html():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" width="device-width, initial-scale=1.0">
        <title>Audio RAG Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, button {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            #status {
                margin-top: 10px;
                padding: 10px;
                border-radius: 4px;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
            }
            .chat-container {
                margin-top: 20px;
                display: none;
            }
            .input-group {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            .input-group input {
                flex: 1;
            }
            .input-group button {
                width: auto;
                margin-top: 0;
                padding: 8px 20px;
            }
            .question-input {
                margin-top: 20px;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }
            #conversation {
                max-height: 400px;
                overflow-y: auto;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-bottom: 20px;
                background-color: #ffffff;
            }
            .message {
                padding: 12px 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                animation: fadeIn 0.3s ease-in;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                position: relative;
                clear: both;
                max-width: 80%;
            }
            .message.user {
                background-color: #e3f2fd;
                margin-left: auto;
                margin-right: 10px;
                border-top-right-radius: 2px;
            }
            .message.assistant {
                background-color: #f5f5f5;
                margin-right: auto;
                margin-left: 10px;
                border-top-left-radius: 2px;
            }
            .message strong {
                display: block;
                margin-bottom: 5px;
                color: #666;
                font-size: 0.9em;
            }
            #loading {
                display: none;
                text-align: center;
                margin: 10px 0;
            }
            footer {
                text-align: center;
                margin-top: 30px;
                color: #666;
                font-size: 0.8em;
            }
        </style>
    </head>
    <body>
        <h1>🎤 Audio RAG Assistant</h1>
        
        <div class="container">
            <div class="form-group">
                <label for="apiKey">GROQ API Key:</label>
                <input type="password" id="apiKey" placeholder="Enter your GROQ API key" required>
            </div>
            
            <div class="form-group">
                <label for="audioFile">Upload Audio File:</label>
                <input type="file" id="audioFile" accept=".mp3,.wav,.m4a,.ogg">
            </div>
            
            <button id="uploadBtn">Process Audio</button>
            
            <div id="status"></div>
            <div id="loading">Processing... please wait</div>
        </div>
        
        <div class="container chat-container" id="chatContainer">
            <div id="conversation"></div>
            
            <div class="form-group question-input">
                <label for="question">Ask a question about the audio:</label>
                <div class="input-group">
                    <input type="text" id="question" placeholder="Type your question here">
                    <button id="askBtn">Ask</button>
                </div>
            </div>
        </div>
        
        <footer>
            Powered by Whisper, Groq, and HuggingFace
        </footer>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const uploadBtn = document.getElementById('uploadBtn');
                const askBtn = document.getElementById('askBtn');
                const statusDiv = document.getElementById('status');
                const loadingDiv = document.getElementById('loading');
                const chatContainer = document.getElementById('chatContainer');
                const conversationDiv = document.getElementById('conversation');
                
                // Add session ID management
                let currentSessionId = null;
                
                uploadBtn.addEventListener('click', async function() {
                    // Reset session when new audio is uploaded
                    currentSessionId = null;
                    // Clear previous conversation
                    conversationDiv.innerHTML = '';
                    
                    const apiKey = document.getElementById('apiKey').value;
                    const fileInput = document.getElementById('audioFile');
                    
                    if (!apiKey) {
                        showStatus('Please enter your GROQ API key', 'error');
                        return;
                    }
                    
                    if (!fileInput.files[0]) {
                        showStatus('Please select an audio file', 'error');
                        return;
                    }
                    
                    // Show loading indicator
                    loadingDiv.style.display = 'block';
                    uploadBtn.disabled = true;
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    formData.append('api_key_param', apiKey);  // Changed from 'api_key' to 'api_key_param'
                    
                    try {
                        const response = await fetch('/upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            showStatus('Audio processed successfully!', 'success');
                            chatContainer.style.display = 'block';
                        } else {
                            showStatus(`Error: ${data.message}`, 'error');
                        }
                    } catch (error) {
                        showStatus(`Error: ${error.message}`, 'error');
                    } finally {
                        loadingDiv.style.display = 'none';
                        uploadBtn.disabled = false;
                    }
                });
                
                askBtn.addEventListener('click', async function() {
                    const question = document.getElementById('question').value;
                    const apiKey = document.getElementById('apiKey').value;
                    
                    if (!question) {
                        return;
                    }
                    
                    // Add user message to conversation
                    addMessage(question, 'user');
                    
                    // Clear input
                    document.getElementById('question').value = '';
                    
                    // Disable button while processing
                    askBtn.disabled = true;
                    
                    try {
                        const response = await fetch('/query', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                question: question,
                                api_key: apiKey,
                                session_id: currentSessionId
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            addMessage(data.answer, 'assistant');
                            // Update session ID
                            currentSessionId = data.session_id;
                        } else {
                            addMessage(`Error: ${data.message}`, 'assistant');
                        }
                    } catch (error) {
                        addMessage(`Error: ${error.message}`, 'assistant');
                    } finally {
                        askBtn.disabled = false;
                    }
                });
                
                function showStatus(message, type) {
                    statusDiv.textContent = message;
                    statusDiv.className = type;
                }
                
                function addMessage(text, role) {
                    const messageDiv = document.createElement('div');
                    messageDiv.classList.add('message', role);
                    
                    const roleLabel = document.createElement('strong');
                    roleLabel.textContent = role === 'user' ? 'You: ' : 'Assistant: ';
                    
                    messageDiv.appendChild(roleLabel);
                    messageDiv.appendChild(document.createTextNode(text));
                    
                    conversationDiv.appendChild(messageDiv);
                    
                    // Scroll to bottom
                    conversationDiv.scrollTop = conversationDiv.scrollHeight;
                }
                
                // Allow Enter key to submit question
                document.getElementById('question').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        askBtn.click();
                    }
                });
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("voice_rag:app", host="0.0.0.0", port=8001, reload=True)