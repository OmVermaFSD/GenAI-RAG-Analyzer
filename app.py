import streamlit as st
import os
import PyPDF2
import google.generativeai as genai

# Simple text splitter
def simple_text_splitter(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Simple vector store using Python lists (no FAISS needed)
class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add_texts(self, texts, embeddings):
        self.chunks.extend(texts)
        self.embeddings.extend(embeddings)
    
    def similarity_search(self, query, k=3):
        # Simple keyword matching as fallback
        results = []
        for i, chunk in enumerate(self.chunks):
            if any(word.lower() in chunk.lower() for word in query.split() if len(word) > 3):
                results.append(chunk)
                if len(results) >= k:
                    break
        return results[:k]

st.set_page_config(page_title="GenAI RAG Analyst", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A;}
    .status-ok {background-color: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 12px; font-weight: bold;}
    .status-err {background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 12px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

def get_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        return None
    return text

def get_expert_response(context, question, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        You are an Expert Strategy Consultant. Answer based ONLY on the Context below.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar
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

# Main UI
st.markdown('<div class="main-header">GenAI Document Analyst</div>', unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "document_text" not in st.session_state: st.session_state.document_text = ""

uploaded_file = st.file_uploader("Upload PDF Report", type="pdf")

if uploaded_file and st.button("🚀 Process Document"):
    if not api_key:
        st.error("❌ API Key Missing.")
    else:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_file)
            if raw_text:
                st.session_state.document_text = raw_text
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
        if st.session_state.document_text:
            with st.spinner("Thinking..."):
                response = get_expert_response(st.session_state.document_text, prompt, api_key)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Please process a document first.")
