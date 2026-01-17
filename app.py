import streamlit as st
import os
import time

# --- 1. BULLETPROOF IMPORTS (Fixes ModuleNotFound) ---
try:
    # Try new LangChain structure first (v0.1+)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError:
    try:
        # Try langchain_core structure
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    except ImportError:
        # Final fallback - try main langchain module
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        except ImportError:
            # If all else fails, use basic imports without text_splitter
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
            # Create a simple text splitter as fallback
            class RecursiveCharacterTextSplitter:
                def __init__(self, chunk_size=1000, chunk_overlap=100):
                    self.chunk_size = chunk_size
                    self.chunk_overlap = chunk_overlap
                
                def split_text(self, text):
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + self.chunk_size
                        chunks.append(text[start:end])
                        start = end - self.chunk_overlap
                    return chunks

import PyPDF2
from langchain.prompts import PromptTemplate

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="GenAI RAG Analyst", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A;}
    .status-ok {background-color: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 12px; font-weight: bold;}
    .status-err {background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 12px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCTIONS ---
def get_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        return None
    return text

def create_vector_store(text_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        if "429" in str(e):
            st.error("⚠️ Free Tier Limit Hit. Please wait 30s and try again.")
        return None

def get_expert_response(vector_store, question, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    template = """
    You are an Expert Strategy Consultant. Answer based ONLY on the Context below.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return chain.run(question)

# --- 4. SIDEBAR & AUTH ---
with st.sidebar:
    st.title("⚙️ Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.markdown('<span class="status-ok">✅ API Key Loaded</span>', unsafe_allow_html=True)
    elif os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.markdown('<span class="status-ok">✅ API Key Loaded</span>', unsafe_allow_html=True)
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if not api_key: st.markdown('<span class="status-err">🔴 Waiting for Key</span>', unsafe_allow_html=True)

# --- 5. MAIN UI ---
st.markdown('<div class="main-header">GenAI Document Analyst</div>', unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

uploaded_file = st.file_uploader("Upload PDF Report", type="pdf")

if uploaded_file and st.button("🚀 Process Document"):
    if not api_key:
        st.error("❌ API Key Missing.")
    else:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_file)
            if raw_text:
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(raw_text)
                vs = create_vector_store(chunks, api_key)
                if vs:
                    st.session_state.vector_store = vs
                    st.success("✅ Analysis Ready!")
            else:
                st.error("❌ Document Empty or Scanned.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if st.session_state.vector_store:
            with st.spinner("Thinking..."):
                response = get_expert_response(st.session_state.vector_store, prompt, api_key)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Please process a document first.")