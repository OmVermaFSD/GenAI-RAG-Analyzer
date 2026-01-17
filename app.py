import streamlit as st
import os
import re
import pandas as pd
import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# --- CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI Pro", page_icon="⚖️", layout="wide")

# --- EXACT IMPORTS (Matched to locked requirements) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import PyPDF2

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
</style>
""", unsafe_allow_html=True)

# --- ENHANCED ANALYSIS FUNCTIONS ---
def extract_legal_entities(text):
    """Extract legal entities like dates, amounts, parties"""
    entities = {
        'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text),
        'amounts': re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', text),
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
        'percentages': re.findall(r'\b\d+(?:\.\d+)?\s*%', text)
    }
    return entities

def analyze_document_risk(text):
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
    """Generate a comprehensive document summary"""
    sentences = sent_tokenize(text)
    word_count = len(word_tokenize(text))
    
    # Extract key information
    entities = extract_legal_entities(text)
    risk_analysis = analyze_document_risk(text)
    
    summary = {
        'total_words': word_count,
        'total_sentences': len(sentences),
        'entities': entities,
        'risk_indicators': risk_analysis,
        'complexity_score': min(100, (word_count / 100) * 10 + len(risk_indicators) * 5)
    }
    
    return summary

def create_risk_chart(risk_indicators):
    """Create a risk visualization chart"""
    if not risk_indicators:
        return None
    
    df = pd.DataFrame(list(risk_indicators.items()), columns=['Risk Factor', 'Count'])
    fig = px.bar(df, x='Risk Factor', y='Count', title='Risk Indicators Analysis',
                  color='Count', color_continuous_scale='Reds')
    return fig

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

def get_legal_response(document_text, question, api_key, analysis_mode="general"):
    """Enhanced legal response with specialized analysis modes"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    
    templates = {
        "general": """
        You are a Senior Legal Analyst. Provide a comprehensive analysis based on the document context.
        
        Context: {context}
        Question: {question}
        
        Analysis:
        """,
        
        "risk": """
        You are a Risk Assessment Specialist. Focus on identifying and explaining potential risks, liabilities, and mitigation strategies.
        
        Context: {context}
        Question: {question}
        
        Risk Analysis:
        """,
        
        "compliance": """
        You are a Compliance Officer. Analyze the document for regulatory compliance issues and requirements.
        
        Context: {context}
        Question: {question}
        
        Compliance Analysis:
        """,
        
        "financial": """
        You are a Financial Legal Analyst. Focus on monetary terms, payment obligations, and financial implications.
        
        Context: {context}
        Question: {question}
        
        Financial Analysis:
        """
    }
    
    template = templates.get(analysis_mode, templates["general"])
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=document_text, question=question)

# --- ENHANCED SIDEBAR ---
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
    
    st.divider()
    
    # Analysis Mode Selection
    st.header("🔍 Analysis Mode")
    analysis_mode = st.selectbox(
        "Choose Analysis Type:",
        ["general", "risk", "compliance", "financial"],
        format_func=lambda x: x.title(),
        help="Select specialized analysis mode for different legal perspectives"
    )
    
    st.divider()
    
    # Quick Actions
    st.header("⚡ Quick Actions")
    if st.session_state.get("document_text"):
        if st.button("📊 Generate Summary"):
            summary = generate_document_summary(st.session_state.document_text)
            st.session_state.document_summary = summary
            st.success("Summary generated!")
        
        if st.button("🎯 Risk Analysis"):
            risk_analysis = analyze_document_risk(st.session_state.document_text)
            st.session_state.risk_analysis = risk_analysis
            st.success("Risk analysis completed!")
    
    st.divider()
    
    # Document Info
    if st.session_state.get("document_text"):
        st.header("📄 Document Info")
        word_count = len(word_tokenize(st.session_state.document_text))
        st.metric("Word Count", word_count)
        st.metric("Characters", len(st.session_state.document_text))

# --- MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI Pro ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Contract Review & Risk Analysis System</div>', unsafe_allow_html=True)

# Display Document Summary if available
if st.session_state.get("document_summary"):
    st.markdown("### 📊 Document Summary")
    summary = st.session_state.document_summary
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Words", summary['total_words'])
    with col2:
        st.metric("Sentences", summary['total_sentences'])
    with col3:
        st.metric("Risk Factors", len(summary['risk_indicators']))
    with col4:
        st.metric("Complexity Score", f"{summary['complexity_score']:.1f}")
    
    # Display entities
    if any(summary['entities'].values()):
        st.markdown("#### 🔍 Key Entities Found")
        entities_cols = st.columns(len(summary['entities']))
        for i, (entity_type, values) in enumerate(summary['entities'].items()):
            with entities_cols[i]:
                st.markdown(f"**{entity_type.title()}:**")
                if values:
                    for val in values[:3]:  # Show first 3
                        st.markdown(f"• {val}")
                    if len(values) > 3:
                        st.markdown(f"... and {len(values) - 3} more")

# Display Risk Analysis if available
if st.session_state.get("risk_analysis"):
    st.markdown("### 🎯 Risk Analysis")
    risk_data = st.session_state.risk_analysis
    
    if risk_data:
        # Create and display risk chart
        fig = create_risk_chart(risk_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display risk details
        with st.expander("📋 Risk Details"):
            for risk, count in risk_data.items():
                st.markdown(f"**{risk.title()}:** Found {count} time(s)")
    else:
        st.info("🎉 No significant risk indicators detected in this document.")

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
                # Auto-generate summary and risk analysis
                summary = generate_document_summary(text)
                st.session_state.document_summary = summary
                risk_analysis = analyze_document_risk(text)
                st.session_state.risk_analysis = risk_analysis
                st.markdown('<div class="success-box">✅ Document Loaded & Analyzed.</div>', unsafe_allow_html=True)
                st.rerun()

# --- CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f"Ask a {analysis_mode} legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if st.session_state.document_text:
        response = get_legal_response(st.session_state.document_text, prompt, api_key, analysis_mode)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and analyze a document first.")
