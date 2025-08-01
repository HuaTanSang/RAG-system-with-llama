"""
This module provides the PDFLoader class for loading PDF documents using LangChain's PyMuPDFLoader.
"""

from langchain_community.document_loaders import PyMuPDFLoader


class PDFLoader:
    """
    PDFLoader wraps LangChain's PyMuPDFLoader to read and load PDF documents.
    """
    def __init__(self):
        """
        Initializes the PDFLoader instance.
        """
        pass

    def read_file(self, file_path):
        """
        Loads a PDF file and returns its contents as a list of Document objects.

        Args:
            file_path (str): Path to the PDF file to be loaded.

        Returns:
            list: List of Document objects containing the PDF's content and metadata.
        """
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        return docs
