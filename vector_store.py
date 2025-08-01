"""
This module provides the VectorStore class for storing and searching document embeddings using FAISS and Ollama embeddings.
"""

import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4



class VectorStore:
    """
    VectorStore manages document embeddings using FAISS for similarity search and Ollama for embedding generation.
    """
    def __init__(self):
        """
        Initializes the VectorStore with Ollama embeddings and a FAISS index.
        """
        # Create an embedding function using Ollama LLM
        self.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        # Initialize FAISS index with the embedding size
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        # Create a FAISS vector store with in-memory docstore
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def add_docs(self, list_of_docs):
        """
        Adds a list of documents to the vector store with unique IDs.

        Args:
            list_of_docs (list): List of Document objects to be embedded and stored.
        """
        uuids = [str(uuid4()) for _ in range(len(list_of_docs))]
        self.vector_store.add_documents(documents=list_of_docs, ids=uuids)

    def search_docs(self, query, k=5):
        """
        Searches for the top-k most similar documents to the query.

        Args:
            query (str): The search query string.
            k (int): Number of top results to return.

        Returns:
            list: List of Document objects most similar to the query.
        """
        results = self.vector_store.similarity_search(
            query,
            k=k
        )
        return results