import streamlit as st
from rag_functions import RAGPipeline
import os
import time

# Set page config
st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
        }
        .stFileUploader>div>div>div>button {
            border-radius: 5px;
        }
        .sidebar .sidebar-content {
            background-color: #e9ecef;
        }
        .reportview-container .markdown-text-container {
            font-family: 'Arial', sans-serif;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize RAG pipeline
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

rag = get_rag_pipeline()

# Sidebar for PDF upload
with st.sidebar:
    st.title("📂 Document Setup")
    st.markdown("Upload a PDF document to build the knowledge base")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document..."):
            # Extract and chunk text
            text = rag.extract_text_from_pdf("temp.pdf")
            chunks = rag.chunk_text(text)
            
            # Create and save index
            index, chunks = rag.create_index(chunks)
            rag.save_index(index, chunks)
            
            # Load the index for use
            rag.load_index()
            
            st.success("Document processed successfully!")
            os.remove("temp.pdf")  # Clean up

# Main content area
st.title("📚 PDF Knowledge Assistant")
st.markdown("Ask questions about your uploaded document and get AI-powered answers")

# Check if index is loaded
if rag.index is None:
    st.warning("Please upload and process a PDF document in the sidebar first.")
    st.stop()

# Query input
query = st.text_input(
    "Enter your question about the document:",
    placeholder="e.g., What is the architecture of DBMS?",
    key="query_input"
)

# Submit button
if st.button("Get Answer", key="get_answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for answers..."):
            start_time = time.time()
            
            # Retrieve relevant chunks
            retrieved_chunks = rag.retrieve(query, k=2)
            
            # Generate answer
            answer = rag.generate_answer(query, retrieved_chunks)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Display results
            st.subheader("Answer")
            st.markdown(f"**{answer}**")
            
            with st.expander("See retrieved context"):
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Context {i}:**")
                    st.info(chunk)
            
            st.caption(f"Response generated in {elapsed_time:.2f} seconds")