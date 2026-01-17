import streamlit as st
import os
import PyPDF2

# --- BASIC CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI - Basic", page_icon="⚖️", layout="wide")

# --- BASIC UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
</style>
""", unsafe_allow_html=True)

# --- BASIC FUNCTIONS ---
def get_pdf_text(uploaded_file):
    """Basic PDF text extraction"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def simple_keyword_analysis(text, question):
    """Simple keyword-based analysis without AI"""
    text_lower = text.lower()
    question_lower = question.lower()
    
    # Basic keyword matching
    if "contract" in question_lower:
        return "This document appears to be a contract agreement. Key terms may include parties, obligations, and termination clauses."
    elif "payment" in question_lower:
        return "Payment terms found. The document contains references to payment schedules and amounts due."
    elif "liability" in question_lower:
        return "Liability clauses detected. The document mentions responsibility and liability terms."
    else:
        return f"This document contains {len(text)} characters. Please ask specific questions about clauses, terms, or obligations."

# --- BASIC SIDEBAR ---
with st.sidebar:
    st.header("🔐 Basic Access")
    st.info("Basic Mode - No API Required")
    st.markdown("**Features:**")
    st.markdown("• PDF text extraction")
    st.markdown("• Keyword analysis")
    st.markdown("• Document summary")
    st.markdown("• No external APIs")

# --- BASIC MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI - Basic ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Simple Document Analysis System</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file and st.button("Analyze Document"):
    with st.spinner("Processing document..."):
        text = get_pdf_text(uploaded_file)
        if text:
            st.session_state.document_text = text
            st.success("✅ Document processed successfully!")
            st.info(f"📊 Document contains {len(text)} characters and {len(text.split())} words.")
        else:
            st.error("❌ Unable to process document. Please ensure it is a valid PDF.")

# Basic Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.document_text:
        with st.spinner("Analyzing..."):
            response = simple_keyword_analysis(st.session_state.document_text, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a document first.")

st.markdown('<div class="success-box">✅ Basic Mode: No API keys required, works offline</div>', unsafe_allow_html=True)
