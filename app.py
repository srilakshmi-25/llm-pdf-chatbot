import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up page configurations
st.set_page_config(
    page_title="Data Engineering Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modular components
from src.pdf_processor import load_all_pdfs_from_directory
from src.chunker import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store, load_vector_store, vector_store_exists
from src.rag_chain import generate_answer
from components.voice_input import voice_input_button

# Cache the embedding model since it is resource-intensive to load on every rerun
@st.cache_resource
def get_cached_embeddings():
    return get_embedding_model()

# Custom CSS for rich premium styling, modern glassmorphism and animations
def inject_custom_css():
    st.markdown("""
        <style>
        /* Base page layout adjustments */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        
        /* Glassmorphism containers */
        .glass-card {
            background: rgba(17, 25, 40, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        
        /* Modern Header styling */
        .title-text {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
            text-shadow: 0 10px 20px rgba(79, 172, 254, 0.15);
        }
        .subtitle-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 1.1rem;
            margin-bottom: 1.8rem;
            font-weight: 300;
        }
        
        /* Chat bubble aesthetics */
        .chat-bubble {
            padding: 1rem 1.25rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            max-width: 85%;
            line-height: 1.5;
            font-size: 0.98rem;
            display: inline-block;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            animation: fadeIn 0.4s ease-out;
        }
        
        .chat-bubble.user {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #f8fafc;
            float: right;
            border-bottom-right-radius: 2px;
        }
        
        .chat-bubble.assistant {
            background: linear-gradient(135deg, #1e1b4b 0%, #111827 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: #f1f5f9;
            float: left;
            border-bottom-left-radius: 2px;
        }
        
        .chat-container {
            width: 100%;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        
        /* Citations styling */
        .citation-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
            margin-bottom: 15px;
            padding-left: 5px;
            width: 100%;
            display: inline-flex;
        }
        
        .citation-tag {
            background: rgba(14, 165, 233, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(14, 165, 233, 0.3);
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 0.78rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 5px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
        }
        
        .citation-tag:hover {
            background: rgba(14, 165, 233, 0.2);
            border-color: #38bdf8;
            transform: translateY(-1px);
        }
        
        /* Status badge styles */
        .status-badge {
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            text-align: center;
            display: inline-block;
        }
        .status-badge.success {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-badge.warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .status-badge.error {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Sidebar Styling overrides */
        .css-163ouus, [data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Inject styling
    inject_custom_css()
    
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Paths
    PDF_DIR = "data/pdfs"
    VECTORSTORE_DIR = "vectorstore"
    
    # Create necessary folders
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # ------------------ SIDEBAR CONFIGURATION ------------------
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #4facfe; margin-bottom: 0;'>⚙️ Bot Control Panel</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 10px; margin-bottom: 20px; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
        
        # 1. API Key Setup
        st.markdown("### 🔑 Groq Authentication")
        # Priority: 1. Sidebar Input, 2. .env file
        env_key = os.environ.get("GROQ_API_KEY", "")
        
        # We mask input by using type="password"
        api_key = st.text_input(
            "Enter Groq API Key:",
            value=env_key if env_key else "",
            type="password",
            placeholder="gsk_...",
            help="Get your free high-speed API Key from console.groq.com"
        )
        
        if api_key:
            st.markdown("<div class='status-badge success'>Groq API Key Configured ✓</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-badge error'>Groq API Key Missing ⚠</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Vector DB Status
        st.markdown("### 📊 Knowledge Database")
        
        # Scan PDFs
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        
        # Load local vector store if it exists and hasn't been loaded in session_state yet
        if st.session_state.vector_store is None:
            if vector_store_exists(VECTORSTORE_DIR):
                try:
                    embeddings = get_cached_embeddings()
                    st.session_state.vector_store = load_vector_store(VECTORSTORE_DIR, embeddings)
                except Exception as e:
                    st.error(f"Failed to load FAISS db: {e}")
                    
        # Render Knowledge DB Status Badge
        if st.session_state.vector_store is not None:
            st.markdown("<div class='status-badge success'>Vector Store: Active ✓</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-badge warning'>Vector Store: Offline (Needs Indexing)</div>", unsafe_allow_html=True)
            
        # Display list of PDFs in directory
        st.markdown(f"**Detected PDFs in `data/pdfs/` ({len(pdf_files)}):**")
        if pdf_files:
            for f in pdf_files:
                st.markdown(f"- 📄 `{f}`")
        else:
            st.info("No PDFs found. Drop some Data Engineering PDF textbooks into the `data/pdfs/` directory to get started.")

        # Re-Index Button
        st.markdown("<br>", unsafe_allow_html=True)
        rebuild_db = st.button("🔄 Ingest & Rebuild Vector Index", use_container_width=True, help="Force rebuild the FAISS database from scratch using all files in data/pdfs/.")
        
        # Ingestion script execution
        if rebuild_db:
            if not pdf_files:
                st.error("No PDFs found to index! Please add PDF textbooks in 'data/pdfs/' first.")
            else:
                with st.spinner("Parsing PDFs and generating embeddings locally (this might take a minute)..."):
                    try:
                        # 1. Parse PDFs
                        raw_docs = load_all_pdfs_from_directory(PDF_DIR)
                        if not raw_docs:
                            st.error("Failed to parse text from the PDF files. Check if they are scanned images or corrupted.")
                        else:
                            # 2. Chunk documents
                            chunks = split_documents(raw_docs)
                            
                            # 3. Load Embeddings Model
                            embeddings = get_cached_embeddings()
                            
                            # 4. Build and save FAISS index
                            st.session_state.vector_store = build_vector_store(chunks, embeddings, VECTORSTORE_DIR)
                            st.success(f"Success! Indexed {len(chunks)} chunks from {len(pdf_files)} PDFs.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Indexing error: {e}")

        # 3. RAG Architecture details
        st.markdown("<br><hr style='border-color: rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)
        with st.expander("ℹ️ System Architecture", expanded=False):
            st.markdown("""
            - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local Execution, CPU)
            - **Vector Index:** `FAISS` (Local Vector Store)
            - **Chunking:** `RecursiveCharacterTextSplitter` (Size: 1500, Overlap: 300)
            - **Retrieval Model:** Top-3 nearest neighbors cosine similarity search
            - **RAG LLM:** `llama-3.1-8b-instant` via Groq
            - **Security:** Strict System Prompts ensuring zero hallucinations and exact citations.
            """)
            
    # ------------------ MAIN INTERFACE ------------------
    st.markdown("<h1 class='title-text'>🤖 Data Engineering Q&A Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Ask questions strictly answered by your Data Engineering textbooks. Hallucination-proof RAG with exact source and page citations.</p>", unsafe_allow_html=True)
    
    # Check if we need to auto-ingest PDFs
    if st.session_state.vector_store is None and pdf_files:
        st.warning("⚡ New PDF files detected in 'data/pdfs/' but no active vector database exists. Auto-building index...")
        with st.spinner("Initializing system and indexing PDFs..."):
            try:
                raw_docs = load_all_pdfs_from_directory(PDF_DIR)
                if raw_docs:
                    chunks = split_documents(raw_docs)
                    embeddings = get_cached_embeddings()
                    st.session_state.vector_store = build_vector_store(chunks, embeddings, VECTORSTORE_DIR)
                    st.success(f"Successfully auto-indexed {len(chunks)} text chunks!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error during auto-indexing: {e}")

    # Display clean welcoming message if conversation is empty
    if not st.session_state.messages:
        st.markdown("""
            <div class='glass-card'>
                <h3>👋 Welcome!</h3>
                <p>This is a strictly grounded Data Engineering Assistant. I extract knowledge solely from textbooks placed inside your <code>data/pdfs/</code> folder.</p>
                <strong>How to get started:</strong>
                <ol>
                    <li>Ensure you have configured your <b>Groq API Key</b> in the sidebar (or in your <code>.env</code> file).</li>
                    <li>Verify that your Data Engineering PDFs are located in <code>data/pdfs/</code> (indicated in the sidebar).</li>
                    <li>Type a query in the text box below or click the microphone button to dictate!</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        citations = msg.get("citations", [])
        
        st.markdown(f"""
            <div class='chat-container'>
                <div class='chat-bubble {role}'>
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display citations beneath the assistant's answer
        if role == "assistant" and citations:
            # We filter out duplicate citations for clean UI representation
            seen_citations = set()
            unique_citations = []
            for cit in citations:
                cit_str = f"{cit['source']}_page_{cit['page']}"
                if cit_str not in seen_citations:
                    seen_citations.add(cit_str)
                    unique_citations.append(cit)
                    
            st.markdown("<div class='citation-container'>", unsafe_allow_html=True)
            for cit in unique_citations:
                source_name = cit["source"]
                page_num = cit["page"]
                st.markdown(f"""
                    <span class='citation-tag' title='Page {page_num}'>
                        📄 {source_name} - Page {page_num}
                    </span>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ VOICE & CHAT INPUT LAYOUT ------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 1. Voice input widget
    # We display it as a floating utility just above the chat input
    voice_input_button(height=60)
    
    # 2. Chat Input field
    # We check if there's a vector store loaded before accepting inputs
    db_active = st.session_state.vector_store is not None
    
    user_query = st.chat_input(
        "Ask a question...",
        disabled=not db_active,
    )
    
    if not db_active:
        st.info("ℹ️ Chat is disabled. Please place Data Engineering PDFs in 'data/pdfs/' and trigger indexing in the sidebar to enable the chatbot.")

    # 3. Process Input
    if user_query and db_active:
        # Append User Message to session state
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()  # Rerun to render user message immediately before LLM call

    # Check if the last message is from user, meaning we need to generate a response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_user_query = st.session_state.messages[-1]["content"]
        
        # Render a spinner while generating answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            citations_placeholder = st.empty()
            
            with st.spinner("Retrieving textbooks and analyzing answer..."):
                if not api_key:
                    ans = "🔑 Groq API Key is missing. Please set it in the sidebar or in your `.env` file to chat."
                    retrieved_docs = []
                else:
                    ans, retrieved_docs = generate_answer(
                        query=latest_user_query,
                        vector_store=st.session_state.vector_store,
                        api_key=api_key,
                        k=3
                    )
                
                # Render response in placeholder
                message_placeholder.markdown(ans)
                
                # Format retrieved docs metadata for citations
                citations = []
                for doc in retrieved_docs:
                    citations.append({
                        "source": doc.metadata.get("source", "Unknown Document"),
                        "page": doc.metadata.get("page", "Unknown Page")
                    })
                
                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans,
                    "citations": citations
                })
                
                # Rerun to display clean styling bubbles and citation tags
                st.rerun()

if __name__ == "__main__":
    main()
