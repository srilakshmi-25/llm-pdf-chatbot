import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document], 
    chunk_size: int = 1500, 
    chunk_overlap: int = 300
) -> List[Document]:
    """
    Splits a list of Documents into chunks using RecursiveCharacterTextSplitter.
    Ensures that page-level and source metadata are preserved for citations.
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []
        
    logger.info(f"Chunking {len(documents)} document pages with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Split completed. Created {len(chunks)} chunks from {len(documents)} pages.")
    
    # Simple validation check to verify metadata is intact
    if chunks and "source" not in chunks[0].metadata:
        logger.warning("Metadata was lost during chunking! Checking text splitter config.")
        
    return chunks
