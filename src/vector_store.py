import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def build_vector_store(
    chunks: List[Document], 
    embeddings: HuggingFaceEmbeddings, 
    save_path: str = "vectorstore"
) -> FAISS:
    """
    Builds a FAISS vector store from document chunks and saves it locally.
    """
    if not chunks:
        raise ValueError("Cannot build vector store: No document chunks provided.")
        
    logger.info(f"Building FAISS vector database from {len(chunks)} chunks...")
    
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        vector_store.save_local(save_path)
        logger.info(f"FAISS vector store successfully saved to '{save_path}'.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error building or saving FAISS vector store: {e}")
        raise e

def load_vector_store(
    save_path: str = "vectorstore", 
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> Optional[FAISS]:
    """
    Loads a locally saved FAISS vector store.
    """
    if not vector_store_exists(save_path):
        logger.warning(f"No FAISS vector store found at '{save_path}'.")
        return None
        
    if embeddings is None:
        from src.embeddings import get_embedding_model
        embeddings = get_embedding_model()
        
    logger.info(f"Loading local FAISS vector store from '{save_path}'...")
    
    try:
        # allow_dangerous_deserialization=True is required to load locally saved FAISS databases
        # as they serialize using Python's pickle library.
        vector_store = FAISS.load_local(
            save_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS vector store successfully loaded.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error loading FAISS vector store from '{save_path}': {e}")
        raise e

def vector_store_exists(save_path: str = "vectorstore") -> bool:
    """
    Checks if a valid FAISS vector store index exists at the specified path.
    FAISS creates two files: index.faiss and index.pkl.
    """
    index_path = os.path.join(save_path, "index.faiss")
    pkl_path = os.path.join(save_path, "index.pkl")
    return os.path.exists(index_path) and os.path.exists(pkl_path)
