import os
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    """
    Parses a PDF file page-by-page using PyMuPDF (fitz).
    Returns a list of LangChain Document objects.
    """
    documents = []
    file_name = os.path.basename(pdf_path)
    
    try:
        import fitz  # PyMuPDF
        logger.info(f"Attempting to parse {file_name} with PyMuPDF...")
        
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text()
            
            # Skip empty pages
            if not text.strip():
                continue
                
            metadata = {
                "source": file_name,
                "page": page_idx + 1,  # 1-indexed page number
                "parser": "PyMuPDF"
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
        logger.info(f"Successfully parsed {len(documents)} pages from {file_name} using PyMuPDF.")
        return documents
        
    except ImportError:
        logger.warning("PyMuPDF is not installed. Falling back to PyPDF2.")
        raise
    except Exception as e:
        logger.error(f"Error parsing {file_name} with PyMuPDF: {e}. Attempting fallback...")
        raise

def parse_pdf_with_pypdf2(pdf_path: str) -> List[Document]:
    """
    Fallback parser using PyPDF2.
    Returns a list of LangChain Document objects.
    """
    documents = []
    file_name = os.path.basename(pdf_path)
    
    try:
        import PyPDF2
        logger.info(f"Parsing {file_name} with fallback PyPDF2...")
        
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_idx in range(len(reader.pages)):
                page = reader.pages[page_idx]
                text = page.extract_text()
                
                # Skip empty pages
                if not text or not text.strip():
                    continue
                    
                metadata = {
                    "source": file_name,
                    "page": page_idx + 1,  # 1-indexed page number
                    "parser": "PyPDF2"
                }
                documents.append(Document(page_content=text, metadata=metadata))
                
        logger.info(f"Successfully parsed {len(documents)} pages from {file_name} using PyPDF2.")
        return documents
        
    except Exception as e:
        logger.error(f"Critical error parsing {file_name} with PyPDF2 fallback: {e}")
        return []

def load_pdf_document(pdf_path: str) -> List[Document]:
    """
    Loads a single PDF document. Tries PyMuPDF first, falls back to PyPDF2 on error.
    """
    try:
        return parse_pdf_with_pymupdf(pdf_path)
    except Exception:
        return parse_pdf_with_pypdf2(pdf_path)

def load_all_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    Scan the specified directory for PDF files and parse each of them.
    Returns a flattened list of all page Documents.
    """
    all_documents = []
    
    if not os.path.exists(directory_path):
        logger.warning(f"Directory '{directory_path}' does not exist. Creating it.")
        os.makedirs(directory_path, exist_ok=True)
        return all_documents
        
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.info(f"No PDF files found in '{directory_path}'. Please add textbooks to this directory.")
        return all_documents
        
    for pdf_file in pdf_files:
        full_path = os.path.join(directory_path, pdf_file)
        docs = load_pdf_document(full_path)
        all_documents.extend(docs)
        
    logger.info(f"Total pages loaded from all PDFs: {len(all_documents)}")
    return all_documents
