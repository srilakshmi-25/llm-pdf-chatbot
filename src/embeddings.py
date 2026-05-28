import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings model locally using sentence-transformers.
    Model: sentence-transformers/all-MiniLM-L6-v2
    
    This operates locally and does not require any external API keys.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Initializing embedding model: {model_name}...")
    
    try:
        # We specify device='cpu' to ensure smooth operation on standard systems without GPU setups.
        # normalize_embeddings=True is set to enable cosine similarity when calculating vector distance.
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise e
