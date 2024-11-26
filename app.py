import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

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
VECTOR_STORE_FILENAME = "faiss_index"
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

    def get_text_chunks(self, text, chunk_size=10000, chunk_overlap=2000):
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        """Create and save vector store"""
        try:
            self.vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            self.vector_store.save_local(VECTOR_STORE_FILENAME)
            st.success("Vector store created successfully!")
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_FILENAME, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return True
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

    # Prepare vector store
    if not os.path.exists(VECTOR_STORE_FILENAME):
        with st.spinner("Processing PDF and creating vector store..."):
            raw_text = chatbot.get_pdf_text(DEFAULT_PDF_PATH)
            if raw_text:
                text_chunks = chatbot.get_text_chunks(raw_text)
                chatbot.create_vector_store(text_chunks)
    else:
        with st.spinner("Loading vector store..."):
            chatbot.load_vector_store()

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