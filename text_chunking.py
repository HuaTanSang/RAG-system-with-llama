"""
This module provides the Chunker class for splitting documents into manageable text chunks using LangChain's RecursiveCharacterTextSplitter.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunker:
    """
    Chunker splits documents into smaller chunks for processing, using customizable chunk size and overlap.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initializes the Chunker with specified chunk size and overlap.

        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n", "\n", " ", ".", ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def chunk_docs(self, docs):
        """
        Splits a list of Document objects into smaller chunks.

        Args:
            docs (list): List of Document objects to be chunked.

        Returns:
            list: List of chunked Document objects with preserved metadata.
        """
        list_of_docs = []
        for doc in docs:
            # Split the document's content into chunks
            tmp = self.text_splitter.split_text(doc.page_content)
            for chunk in tmp:
                list_of_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata,
                    )
                )
        return list_of_docs
