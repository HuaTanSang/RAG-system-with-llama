# RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, and Ollama LLMs. It processes PDF documents, chunks their text, stores embeddings in a vector store, and answers user queries based on document context.

## Features
- Load PDF documents and extract text
- Chunk text for efficient retrieval
- Store and search document embeddings using FAISS
- Generate answers using Ollama LLM, strictly based on retrieved context

## File Overview
- `main.py`: Entry point; runs the RAG pipeline on a sample PDF and query
- `rag.py`: Orchestrates PDF loading, chunking, vector storage, and LLM response
- `pdfloader.py`: Loads PDF files using LangChain's PyMuPDFLoader
- `text_chunking.py`: Splits documents into manageable text chunks
- `vector_store.py`: Stores and searches document embeddings using FAISS
- `requirements.txt`: Python dependencies
- `data.pdf`: Sample PDF document

## Usage
1. Clone and cd to this repository: 
    ```bash
    git clone https://github.com/HuaTanSang/RAG-system-with-llama.git && 
    cd RAG-system-with-llama
    ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open another terminal and start LLAMA model 
    ```bash
    # Download ollama 
    curl -fsSL https://ollama.com/install.sh | sh  
    # Pull and run llama model (this project use llama3.2:1b)
    ollama pull llama3.2:1b && ollama run llama3.2:1b
    ```

4. Place your PDF file in the project directory (default: `data.pdf`).
5. Run the main script:
   ```bash
   python3 main.py
   ```
6. The system will print the answer to the sample query based on the PDF content. You can change query in the `main.py` file. 

## Configuration
- Model name can be set in `.env` (default: `llama3.2:1b`)
- Chunk size and overlap can be adjusted in `text_chunking.py`

## Requirements
- Ubuntu 22.04.5 LTS 
- Python 3.10+ (conda is recommended)
- Ollama (for LLM inference)