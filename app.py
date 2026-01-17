import streamlit as st
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI", page_icon="⚖️", layout="wide")

# --- EXACT IMPORTS (Matched to locked requirements) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import PyPDF2

# --- UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
</style>
""", unsafe_allow_html=True)

# --- CORE FUNCTIONS ---
def get_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        return None
    return text

def get_legal_response(document_text, question, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    template = """
    You are a Senior Legal Analyst. Answer based on the context.
    If not found, say "Not found in document."
    Context: {context}
    Question: {question}
    Analysis:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=document_text, question=question)

# --- SIDEBAR WITH API KEY ---
with st.sidebar:
    st.header("🔐 Secure Access")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Credentials Authenticated")
    elif os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.success("Credentials Authenticated")
    else:
        api_key = st.text_input("Enter API Key", type="password")

# --- MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Contract Review & Risk Analysis System</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None

uploaded_file = st.file_uploader("Upload Legal Agreement (PDF)", type="pdf")

if uploaded_file and st.button("Analyze Document"):
    if not api_key:
        st.markdown('<div class="error-box">⚠️ Please provide an API Key.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Processing..."):
            text = get_pdf_text(uploaded_file)
            if text:
                st.session_state.document_text = text
                st.markdown('<div class="success-box">✅ Document Loaded.</div>', unsafe_allow_html=True)

# --- CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if st.session_state.document_text:
        response = get_legal_response(st.session_state.document_text, prompt, api_key)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and analyze a document first.")
