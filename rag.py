"""
This module provides the RAG class, which orchestrates the Retrieval-Augmented Generation pipeline using LangChain, FAISS, and Ollama LLMs.
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_ollama import OllamaLLM

from pdfloader import PDFLoader
from vector_store import VectorStore
from text_chunking import Chunker


class RAG:
    """
    RAG orchestrates the Retrieval-Augmented Generation pipeline:
    - Loads PDF documents
    - Chunks text
    - Stores and searches embeddings
    - Generates answers using an LLM strictly based on retrieved context
    """
    def __init__(self):
        """
        Initializes the RAG pipeline components and loads environment variables.
        """
        load_dotenv()

        # Instruction prompt for the LLM to ensure context-based answers
        self.instructor_prompt = (
            "Instruction: You're an expert problem solver you answer questions from context given below. "
            "You strictly adhere to the context and never move away from it. "
            "You're honest and if you do not find the answer to the question in the context you politely say 'I Don't know!'\n"
            "So help me answer the user question mentioned below with the help of the context provided\n"
            "User Question: {user_query}\n"
            "Answer Context: {answer_context}\n"
        )
        # Get model name from environment or use default
        model_name = os.getenv("MODEL_NAME", "llama3.2:1b")

        # Initialize LangChain components
        self.prompt = PromptTemplate.from_template(self.instructor_prompt)
        self.llm = OllamaLLM(model=model_name)
        self.vector_store = VectorStore()
        self.pdfloader = PDFLoader()
        self.chunker = Chunker()

    def run(self, filePath, query):
        """
        Runs the RAG pipeline: loads, chunks, stores, searches, and answers a query.

        Args:
            filePath (str): Path to the PDF file to process.
            query (str): User's question to answer.

        Returns:
            str: LLM-generated answer strictly based on retrieved context.
        """
        # Load PDF and extract documents
        docs = self.pdfloader.read_file(file_path=filePath)
        # Chunk documents for efficient retrieval
        list_of_docs = self.chunker.chunk_docs(docs)
        # Add chunks to vector store
        self.vector_store.add_docs(list_of_docs)
        # Search for relevant chunks based on query
        results = self.vector_store.search_docs(query)
        answer_context = "\n\n"

        # Aggregate retrieved chunk contents
        for res in results:
            answer_context = answer_context + "\n\n" + res.page_content

        # Create a chain with prompt and LLM
        chain = self.prompt | self.llm
        # Invoke the chain with user query and context
        response = chain.invoke(
            {
                "user_query": query,
                "answer_context": answer_context,
            }
        )
        return response