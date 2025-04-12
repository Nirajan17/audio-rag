import re
import os
import time
import streamlit as st
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Page configuration for a cleaner look
st.set_page_config(
    page_title="Audio RAG Assistant",
    page_icon="🔊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .transcript-container {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #000000;
    }
    .assistant-message {
        background-color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# Directory setup
audios_directory = 'audio-files/'
os.makedirs(audios_directory, exist_ok=True)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# LLM template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Initialize HuggingFace embeddings
@st.cache_resource
def load_embeddings():
    with st.spinner("Loading embedding model..."):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs
        )

# Initialize Groq LLM
@st.cache_resource
def load_llm():
    with st.spinner("Connecting to Groq..."):
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=API_KEY,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    with st.spinner("Loading Whisper model..."):
        return whisper.load_model("medium.en")

# Initialize vector store
@st.cache_resource
def init_vector_store(_embeddings):
    return InMemoryVectorStore(_embeddings)

# Load required models
embeddings = load_embeddings()
vector_store = init_vector_store(embeddings)
model = load_llm()
whisper_model = load_whisper_model()

# Function to upload audio
def upload_audio(file):
    file_path = os.path.join(audios_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# Function to transcribe audio with progress
def transcribe_audio(file_path):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Analyzing audio...")
    progress_bar.progress(25)
    time.sleep(0.5)
    
    status_text.text("Transcribing speech...")
    progress_bar.progress(50)
    
    result = whisper_model.transcribe(file_path)
    
    status_text.text("Processing transcript...")
    progress_bar.progress(75)
    time.sleep(0.5)
    
    progress_bar.progress(100)
    status_text.text("Transcription complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return result["text"]

# Function to split text
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

# Function to index documents with progress
def index_docs(texts):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Creating embeddings...")
    progress_bar.progress(30)
    time.sleep(0.5)
    
    vector_store.add_texts(texts)
    
    status_text.text("Building vector index...")
    progress_bar.progress(70)
    time.sleep(0.5)
    
    progress_bar.progress(100)
    status_text.text("Vector index complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.vector_store_ready = True

# Function to retrieve documents
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Function to answer questions
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"question": question, "context": context})
    return clean_text(response)

# Function to clean text
def clean_text(text):
    cleaned_text = text
    return cleaned_text

# Main app layout
st.title("🎤 Audio RAG Assistant")
st.markdown("Upload an audio file, ask questions about its content, and get AI-powered answers.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav", "m4a", "ogg"],
        accept_multiple_files=False,
        key="audio_uploader"
    )

if uploaded_file and (st.session_state.last_uploaded_file != uploaded_file.name):
    st.session_state.last_uploaded_file = uploaded_file.name
    st.session_state.vector_store_ready = False
    st.session_state.transcript = ""
    
    with st.spinner("Processing audio file..."):
        file_path = upload_audio(uploaded_file)
        transcript = transcribe_audio(file_path)
        st.session_state.transcript = transcript
        
        chunked_texts = split_text(transcript)
        index_docs(chunked_texts)
    
    st.success(f"✅ File '{uploaded_file.name}' processed successfully!")

# Display transcript if available
# if st.session_state.transcript:
#     with col2:
#         st.subheader("📝 Transcript")
        # st.markdown(f"<div class='transcript-container'>{st.session_state.transcript}</div>", unsafe_allow_html=True)

# Display chat interface only if vector store is ready
if st.session_state.vector_store_ready:
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message assistant-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    question = st.chat_input("Ask a question about the audio...", disabled=not st.session_state.vector_store_ready)
    
    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display the new user message (without rerunning)
        st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {question}</div>", unsafe_allow_html=True)
        
        # Get and display answer
        with st.spinner("Thinking..."):
            related_docs = retrieve_docs(question)
            answer = answer_question(question, related_docs)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer.content})
        
        # Display the new assistant message (without rerunning)
        st.markdown(f"<div class='chat-message assistant-message'><strong>Assistant:</strong> {answer.content}</div>", unsafe_allow_html=True)
        
        # Force a rerun to update the UI
        st.rerun()
else:
    if uploaded_file:
        st.info("Processing audio file... Please wait.")
    else:
        st.info("Upload an audio file to begin.")

# Footer
st.markdown("---")
st.caption("Powered by Whisper, Groq, and HuggingFace")