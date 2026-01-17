import streamlit as st
import os
import PyPDF2

# --- AI CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI - AI Mode", page_icon="🤖", layout="wide")

# --- AI UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
</style>
""", unsafe_allow_html=True)

# --- AI IMPORTS (ISOLATED) ---
try:
    from langchain.chains import LLMChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    AI_AVAILABLE = True
    st.success("✅ AI Components Loaded")
except ImportError as e:
    AI_AVAILABLE = False
    st.error(f"❌ AI Import Error: {e}")

# --- AI FUNCTIONS ---
def get_pdf_text(uploaded_file):
    """PDF text extraction"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def get_ai_response(document_text, question, api_key):
    """AI-powered legal analysis"""
    if not AI_AVAILABLE:
        return "AI components not available. Please check imports."
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
        template = """
        You are a Senior Legal Analyst. Analyze the document context.
        Context: {context}
        Question: {question}
        Analysis:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(context=document_text, question=question)
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- AI SIDEBAR ---
with st.sidebar:
    st.header("🤖 AI Access")
    if AI_AVAILABLE:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("Credentials Loaded")
        elif os.getenv("GOOGLE_API_KEY"):
            api_key = os.getenv("GOOGLE_API_KEY")
            st.success("Credentials Loaded")
        else:
            api_key = st.text_input("Enter API Key", type="password")
    else:
        st.error("AI Components Not Available")
        api_key = None

# --- AI MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI - AI Mode 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Document Analysis</div>', unsafe_allow_html=True)

if AI_AVAILABLE:
    st.success("✅ AI Mode: Advanced analysis with Gemini AI")
else:
    st.error("❌ AI Mode: Components not loaded")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file and st.button("Analyze with AI"):
    if not AI_AVAILABLE:
        st.error("❌ AI components not available")
    elif not api_key:
        st.error("❌ Please provide an API Key")
    else:
        with st.spinner("AI Processing..."):
            text = get_pdf_text(uploaded_file)
            if text:
                st.session_state.document_text = text
                st.success("✅ Document ready for AI analysis")
            else:
                st.error("❌ Unable to process document")

# AI Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask AI about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.document_text and AI_AVAILABLE and api_key:
        with st.spinner("AI Thinking..."):
            response = get_ai_response(st.session_state.document_text, prompt, api_key)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload document and ensure AI is available")

st.markdown('<div class="error-box">⚠️ AI Mode: Requires API key and working imports</div>', unsafe_allow_html=True)
