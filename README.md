# RAG Pipeline for Document Retrieval and Generation

A lightweight Retrieval-Augmented Generation (RAG) pipeline for extracting, indexing, and querying PDF documents using FAISS for semantic search and Ollama (LLaMA 2) for answer generation.

![RAG Workflow](https://tse4.mm.bing.net/th/id/OIP.3TfftXB_HV7CRm8Ir1LmAwHaEa?r=0&rs=1&pid=ImgDetMain&o=7&rm=3)

## Features

- **PDF Text Extraction**: Extract raw text from PDFs using PyMuPDF
- **Semantic Chunking**: Split documents into meaningful chunks for retrieval
- **FAISS Indexing**: Create efficient vector indices for fast similarity search
- **Ollama Integration**: Generate answers using LLaMA 2 with retrieved context
- **Modular Design**: Use as a library (`rag_functions.py`) or standalone script (`main.py`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dhruv-Gadre/basic_rag
   cd basic_rag
    ```

2. **Install Dependencies**
    ```bash
    pip install requirements.txt
    ```

3. **Setup Ollama**
    ```bash
    ollama pull llama2:7b
    ollama run llama2:7b
    ```

## Usage

- After you have done the above steps
- Go to the cmd prompt and run "streamlit app.py"

## Et Voila, It's Done