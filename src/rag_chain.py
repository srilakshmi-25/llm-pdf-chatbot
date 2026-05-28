import logging
import time
from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from src.prompts import RAG_PROMPT

logger = logging.getLogger(__name__)

def retrieve_top_k(vector_store: FAISS, query: str, k: int = 3) -> List[Document]:
    """
    Retrieves the top K most relevant document chunks from the FAISS database.
    """
    logger.info(f"Retrieving top-{k} relevant chunks for query: '{query}'")
    try:
        # Perform similarity search
        docs = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents from FAISS: {e}")
        raise e

def format_context_with_metadata(docs: List[Document]) -> str:
    """
    Formats retrieved documents into a clean context string,
    clearly identifying the filename and page number for each chunk.
    This assists the LLM in citing sources correctly.
    """
    context_parts = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")
        content = doc.page_content.strip()
        
        chunk_text = (
            f"[Chunk #{idx + 1}]\n"
            f"Source Filename: {source}\n"
            f"Page Number: {page}\n"
            f"Text Content:\n{content}\n"
            f"------------------------"
        )
        context_parts.append(chunk_text)
        
    return "\n\n".join(context_parts)

def generate_answer(
    query: str, 
    vector_store: FAISS, 
    api_key: str, 
    k: int = 3
) -> Tuple[str, List[Document]]:
    """
    Orchestrates the complete RAG pipeline:
    1. Retrieve the top K chunks.
    2. Format the context with precise source metadata.
    3. Invoke Llama-3.1-8b-instant on Groq.
    4. Handles API errors (like 429 rate limits) gracefully.
    
    Returns a tuple containing:
      - The generated response (string)
      - The list of retrieved source documents (for UI citation rendering)
    """
    if not api_key:
        raise ValueError("Groq API Key is missing. Please set it in your .env file or the sidebar.")
        
    # Step 1: Retrieve context
    try:
        retrieved_docs = retrieve_top_k(vector_store, query, k=k)
    except Exception as e:
        return f"Error querying vector database: {str(e)}", []
        
    if not retrieved_docs:
        return "I could not find this in the provided textbooks.", []
        
    # Step 2: Format context
    formatted_context = format_context_with_metadata(retrieved_docs)
    
    # Step 3: Run LLM Chain with Groq
    max_retries = 3
    retry_delay = 2.0  # seconds
    
    for attempt in range(max_retries):
        try:
            # Initialize ChatGroq LLM
            # Temperature = 0.0 forces the model to be highly deterministic and factual.
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=api_key,
                temperature=0.0,
                max_retries=2
            )
            
            # Format prompt with context and query
            prompt_input = {
                "context": formatted_context,
                "question": query
            }
            formatted_prompt = RAG_PROMPT.format(**prompt_input)
            
            logger.info("Sending request to Groq API...")
            start_time = time.time()
            
            # Invoke LLM
            response = llm.invoke(formatted_prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"Received Groq response in {elapsed:.2f}s.")
            
            return response.content.strip(), retrieved_docs
            
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_str}")
            
            # Handle standard rate limit errors (HTTP 429)
            if "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Groq API Rate Limit (429) hit. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2.0  # Exponential backoff
                    continue
                else:
                    return (
                        "⚠️ Groq API Rate Limit exceeded. We hit too many requests in a short time. "
                        "Please wait a minute and try again.", 
                        retrieved_docs
                    )
            
            # General authentication errors (HTTP 401)
            elif "401" in error_str or "unauthorized" in error_str.lower() or "api_key" in error_str.lower():
                return (
                    "🔑 Invalid Groq API Key! Please verify that your GROQ_API_KEY is correct "
                    "in the sidebar or in your .env file.",
                    []
                )
                
            # Other general connection errors
            else:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return f"An error occurred while connecting to Groq API: {error_str}", retrieved_docs
                
    return "Failed to get response from AI after multiple retries due to server overload.", retrieved_docs
