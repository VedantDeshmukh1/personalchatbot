import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
import hashlib
from pathlib import Path

# PDF and Text Processing
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM and Chains
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()

# Configuration Constants
DEFAULT_PDF_PATH = "data/about.pdf"
VECTOR_STORE_DIR = "vector_store"
PDF_HASH_FILE = os.path.join(VECTOR_STORE_DIR, "pdf_hash.txt")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

class PDFChatbot:
    def __init__(self):
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")
            raise

        # Initialize vector store
        self.vector_store = None

        # Create vector store directory if it doesn't exist
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    def get_pdf_text(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            return text
        except FileNotFoundError:
            st.error(f"PDF file not found at {pdf_path}")
            return ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    def get_text_chunks(self, text, chunk_size=2000, chunk_overlap=500):
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    def get_pdf_hash(self, pdf_path):
        """Calculate MD5 hash of PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                return hashlib.md5(file.read()).hexdigest()
        except Exception as e:
            st.error(f"Error calculating PDF hash: {e}")
            return None

    def save_pdf_hash(self, pdf_hash):
        """Save PDF hash to file"""
        try:
            with open(PDF_HASH_FILE, 'w') as file:
                file.write(pdf_hash)
        except Exception as e:
            st.error(f"Error saving PDF hash: {e}")

    def get_saved_pdf_hash(self):
        """Get saved PDF hash"""
        try:
            if os.path.exists(PDF_HASH_FILE):
                with open(PDF_HASH_FILE, 'r') as file:
                    return file.read().strip()
        except Exception as e:
            st.error(f"Error reading PDF hash: {e}")
        return None

    def create_vector_store(self, text_chunks):
        """Create and save vector store"""
        try:
            self.vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            self.vector_store.save_local(VECTOR_STORE_DIR)
            st.success("Vector store created successfully!")
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            if os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_DIR, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            return False
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")
            return False

    def get_conversational_chain(self):
        """Create conversational QA chain"""
        prompt_template = """
        Context: {context}
        
        Question: {question}
        
        Instructions:
        - Answer based strictly on the provided context
        - If the answer isn't in the context, respond with: "I don't have this information in my knowledge base, but I'll be happy to check with Vedant and get back to you on this!"
        - Provide a clear, concise, and structured response
        - If asked "Who are you?", respond "I am Vedant's personal assistant chatbot"
        - Maintain a friendly and helpful tone
        
        Answer:
        """
        
        try:
            model = ChatGroq(
                temperature=0.3, 
                model_name=LLM_MODEL,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        except Exception as e:
            st.error(f"Failed to create conversational chain: {e}")
            return None

    def process_query(self, query):
        """Process user query and retrieve response"""
        if not self.vector_store:
            st.warning("Vector store not loaded. Please process PDF first.")
            return "I'm unable to process your query at the moment."
        
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=3)
            
            # Get conversational chain
            chain = self.get_conversational_chain()
            
            if not chain:
                return "Sorry, I'm experiencing technical difficulties."
            
            # Get response
            response = chain(
                {"input_documents": docs, "question": query}, 
                return_only_outputs=True
            )
            
            return response.get("output_text", "I couldn't generate a response.")
        
        except Exception as e:
            st.error(f"Query processing error: {e}")
            return "I encountered an error processing your query."

def main():
    # Streamlit UI Configuration
    st.set_page_config(page_title="Vedant's Assistant", page_icon="🤖")
    st.title("💬 Chat with Vedant's AI Assistant")

    # Initialize chatbot
    chatbot = PDFChatbot()

    # Check if PDF exists
    if not os.path.exists(DEFAULT_PDF_PATH):
        st.error(f"PDF file not found at {DEFAULT_PDF_PATH}")
        return

    # Get current PDF hash
    current_pdf_hash = chatbot.get_pdf_hash(DEFAULT_PDF_PATH)
    saved_pdf_hash = chatbot.get_saved_pdf_hash()

    # Check if vector store needs to be created or updated
    vector_store_exists = os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss"))
    
    if not vector_store_exists or current_pdf_hash != saved_pdf_hash:
        with st.spinner("Processing PDF and creating vector store..."):
            raw_text = chatbot.get_pdf_text(DEFAULT_PDF_PATH)
            if raw_text:
                text_chunks = chatbot.get_text_chunks(raw_text)
                chatbot.create_vector_store(text_chunks)
                chatbot.save_pdf_hash(current_pdf_hash)
                st.success("Vector store updated successfully!")
    else:
        with st.spinner("Loading existing vector store..."):
            if not chatbot.load_vector_store():
                st.error("Failed to load vector store. Please refresh the page.")
                return

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Vedant's AI assistant. Ask me anything about him!"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if prompt := st.chat_input("Ask a question about Vedant"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = chatbot.process_query(prompt)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
