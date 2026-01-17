import streamlit as st
import os

# --- PAGE CONFIGURATION (Legal/Corporate Look) ---
st.set_page_config(
    page_title="Legal Insight AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BULLETPROOF IMPORTS ---
# This fixes specific cloud error you have been seeing
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

import PyPDF2
from langchain.prompts import PromptTemplate

# --- PROFESSIONAL CSS ---
st.markdown("""
<style>
    .main-header {font-family: 'serif'; font-size: 2.5rem; color: #0f172a; border-bottom: 2px solid #cbd5e1; margin-bottom: 1rem;}
    .sub-text {font-family: 'sans-serif'; color: #64748b; font-size: 1.1rem;}
    .disclaimer {font-size: 0.8rem; color: #94a3b8; margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 10px;}
    .stChatMessage {background-color: #f8fafc; border: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        return None
    return text

def get_response(vector_store, question, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    
    # Legal-Specific Prompt
    template = """
    You are a Senior Legal Analyst. Analyze the provided contract/document context strictly.
    
    Rules:
    1. Cite specific clauses or sections if available.
    2. Highlight potential risks in **bold**.
    3. If the answer is not in the document, state: "Not found in the provided text."
    4. Maintain a neutral, professional tone.
    
    Context: {context}
    Question: {question}
    
    Analysis:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return chain.run(question)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Legal AI Suite")
    st.info("Strictly Confidential\nClient-Attorney Privilege Level Security")
    
    # Auto-Load Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Credentials Loaded")
    elif os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.success("Credentials Loaded")
    else:
        api_key = st.text_input("Enter License Key", type="password")

# --- MAIN APP ---
st.markdown('<div class="main-header">Contract & Legal Document Analyst</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload legal agreements for instant risk assessment and clause extraction.</p>', unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

uploaded_file = st.file_uploader("Upload PDF Agreement", type="pdf")

if uploaded_file and st.button("Analyze Document"):
    if not api_key:
        st.error("Authentication Missing: Please provide an API Key.")
    else:
        with st.spinner("Scanning document for clauses..."):
            text = get_pdf_text(uploaded_file)
            if text:
                # Create Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
                st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.success("Document Indexed Successfully")
            else:
                st.error("Unable to read document. Ensure it is not a scanned image.")

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about liability, termination clauses, or payment terms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    if st.session_state.vector_store:
        with st.spinner("Consulting..."):
            response = get_response(st.session_state.vector_store, prompt, api_key)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a document first.")

st.markdown('<div class="disclaimer">Disclaimer: This tool is an AI assistant and does not constitute professional legal advice. Verify all outputs with qualified counsel.</div>', unsafe_allow_html=True)
