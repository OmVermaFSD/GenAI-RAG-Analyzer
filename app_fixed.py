import streamlit as st
import os
import re
import PyPDF2

# --- FIXED CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI - Fixed", page_icon="⚖️", layout="wide")

# --- FIXED UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
</style>
""", unsafe_allow_html=True)

# --- FIXED IMPORTS (NO NLTK) ---
try:
    from langchain.chains import LLMChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    AI_AVAILABLE = True
    st.success("✅ AI Components Loaded")
except ImportError as e:
    AI_AVAILABLE = False
    st.error(f"❌ AI Import Error: {e}")

# --- FIXED FUNCTIONS (NO NLTK) ---
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

def simple_sentence_split(text):
    """Simple sentence splitting without NLTK"""
    # Split by common sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Clean up empty strings and whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def simple_word_count(text):
    """Simple word counting without NLTK"""
    # Split by whitespace and filter out empty strings
    words = [word.strip() for word in text.split() if word.strip()]
    return len(words)

def extract_entities(text):
    """Extract legal entities without NLTK"""
    entities = {
        'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text),
        'amounts': re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', text),
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
        'percentages': re.findall(r'\b\d+(?:\.\d+)?\s*%', text)
    }
    return entities

def analyze_risk(text):
    """Analyze document for risk indicators"""
    risk_keywords = [
        'liability', 'terminate', 'breach', 'penalty', 'damage', 'forfeit',
        'sue', 'lawsuit', 'arbitration', 'indemnify', 'warranty', 'guarantee'
    ]
    
    risk_indicators = {}
    for keyword in risk_keywords:
        count = len(re.findall(rf'\b{keyword}\b', text, re.IGNORECASE))
        if count > 0:
            risk_indicators[keyword] = count
    
    return risk_indicators

def generate_document_summary(text):
    """Generate document summary without NLTK"""
    # Use simple text processing
    sentences = simple_sentence_split(text)
    word_count = simple_word_count(text)
    
    # Extract key information
    entities = extract_entities(text)
    risk_analysis = analyze_risk(text)
    
    summary = {
        'total_words': word_count,
        'total_sentences': len(sentences),
        'entities': entities,
        'risk_indicators': risk_analysis,
        'complexity_score': min(100, (word_count / 100) * 10 + len(risk_analysis) * 5)
    }
    
    return summary

def get_ai_response(document_text, question, api_key):
    """AI-powered legal analysis"""
    if not AI_AVAILABLE:
        return "AI components not available. Please check imports."
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
        template = """
        You are a Senior Legal Analyst. Analyze document context.
        Context: {context}
        Question: {question}
        Analysis:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(context=document_text, question=question)
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- FIXED SIDEBAR ---
with st.sidebar:
    st.header("🔐 Fixed Access")
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

# --- FIXED MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI - Fixed ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">No NLTK Dependencies - Always Works</div>', unsafe_allow_html=True)

if AI_AVAILABLE:
    st.success("✅ Fixed Mode: AI components working")
else:
    st.error("❌ Fixed Mode: AI components not loaded")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file and st.button("Fixed Analysis"):
    with st.spinner("Processing..."):
        text = get_pdf_text(uploaded_file)
        if text:
            st.session_state.document_text = text
            # Generate summary without NLTK
            summary = generate_document_summary(text)
            st.session_state.document_summary = summary
            st.success("✅ Document processed successfully!")
            
            # Display summary
            st.markdown("### 📊 Document Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Words", summary['total_words'])
            with col2:
                st.metric("Sentences", summary['total_sentences'])
            with col3:
                st.metric("Risk Factors", len(summary['risk_indicators']))
            with col4:
                st.metric("Complexity", f"{summary['complexity_score']:.1f}")
            
            # Display entities
            if any(summary['entities'].values()):
                st.markdown("#### 🔍 Key Entities")
                for entity_type, values in summary['entities'].items():
                    if values:
                        st.markdown(f"**{entity_type.title()}:** {', '.join(values[:3])}")
            
            # Display risk analysis
            if summary['risk_indicators']:
                st.markdown("#### 🎯 Risk Analysis")
                with st.expander("Risk Details"):
                    for risk, count in summary['risk_indicators'].items():
                        st.markdown(f"**{risk.title()}:** {count} occurrence(s)")
            else:
                st.info("🎉 No significant risk indicators detected.")
        else:
            st.error("❌ Unable to process document")

# Fixed Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.document_text and AI_AVAILABLE and api_key:
        with st.spinner("AI Thinking..."):
            response = get_ai_response(st.session_state.document_text, prompt, api_key)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    elif st.session_state.document_text:
        # Fallback to simple analysis
        response = f"Document contains {len(st.session_state.document_text)} characters. Please provide API key for AI analysis."
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a document first.")

st.markdown('<div class="success-box">✅ Fixed Mode: No NLTK dependencies, works on all environments</div>', unsafe_allow_html=True)
