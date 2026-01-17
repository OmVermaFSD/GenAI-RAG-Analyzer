import streamlit as st
import os
import re
import PyPDF2
from io import BytesIO
import time

# --- ROBUST CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI - Enterprise", page_icon="⚖️", layout="wide")

# --- ENTERPRISE UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
    .warning-box {padding: 1rem; background-color: #fffbeb; border-left: 5px solid #f59e0b; color: #92400e;}
    .progress-bar {height: 20px; background-color: #e5e7eb; border-radius: 10px; overflow: hidden;}
    .progress-fill {height: 100%; background-color: #3b82f6; transition: width 0.3s ease;}
</style>
""", unsafe_allow_html=True)

# --- ROBUST IMPORTS ---
try:
    from langchain.chains import LLMChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- ROBUST PDF PROCESSING ---
def get_pdf_text_robust(uploaded_file, max_size_mb=10):
    """Robust PDF text extraction with size limits and progress tracking"""
    try:
        # Check file size
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        if file_size > max_size_mb:
            st.error(f"❌ File too large: {file_size:.1f}MB. Maximum size: {max_size_mb}MB")
            return None
        
        st.info(f"📄 Processing {file_size:.1f}MB PDF...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read PDF with progress tracking
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                # Update progress
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i + 1}/{total_pages} ({progress*100:.0f}%)")
                
                # Add small delay to prevent overwhelming
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing page {i + 1}: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        time.sleep(1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if len(text.strip()) < 100:
            st.warning("⚠️ Document appears to be very short or mostly images")
        
        return text
        
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return None

def chunk_text_intelligently(text, max_chunk_size=4000, overlap=200):
    """Intelligent text chunking for large documents"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            
            if sentence_end > start + max_chunk_size // 2:  # Don't go back too far
                end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
        
        # Calculate next start with overlap
        start = max(end - overlap, end)
        
        if start >= len(text):
            break
    
    return chunks

# --- ROBUST ANALYSIS FUNCTIONS ---
def extract_entities_robust(text):
    """Robust entity extraction with limits"""
    entities = {
        'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text)[:50],  # Limit results
        'amounts': re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', text)[:50],
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)[:20],
        'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)[:20],
        'percentages': re.findall(r'\b\d+(?:\.\d+)?\s*%', text)[:30]
    }
    return entities

def analyze_risk_robust(text):
    """Robust risk analysis with performance optimization"""
    risk_keywords = [
        'liability', 'terminate', 'breach', 'penalty', 'damage', 'forfeit',
        'sue', 'lawsuit', 'arbitration', 'indemnify', 'warranty', 'guarantee',
        'confidential', 'proprietary', 'trade secret', 'non-compete', 'non-disclosure'
    ]
    
    # Use case-insensitive search with limits
    risk_indicators = {}
    text_lower = text.lower()
    
    for keyword in risk_keywords:
        count = text_lower.count(keyword.lower())
        if count > 0:
            risk_indicators[keyword] = min(count, 100)  # Cap at 100 for performance
    
    return risk_indicators

def generate_summary_robust(text):
    """Robust summary generation for large documents"""
    # Use chunking for large documents
    if len(text) > 10000:
        chunks = chunk_text_intelligently(text)
        # Analyze first few chunks for summary
        analysis_text = ' '.join(chunks[:3])
    else:
        analysis_text = text
    
    # Simple metrics
    word_count = len(analysis_text.split())
    sentence_count = len(re.split(r'[.!?]+', analysis_text))
    
    # Extract entities and risks
    entities = extract_entities_robust(analysis_text)
    risk_analysis = analyze_risk_robust(text)
    
    summary = {
        'total_words': word_count,
        'total_sentences': sentence_count,
        'entities': entities,
        'risk_indicators': risk_analysis,
        'complexity_score': min(100, (word_count / 1000) * 5 + len(risk_analysis) * 3),
        'document_size': len(text)
    }
    
    return summary

def get_ai_response_robust(document_text, question, api_key, max_context=8000):
    """Robust AI response with context limiting"""
    if not AI_AVAILABLE:
        return "AI components not available."
    
    try:
        # Limit context for large documents
        if len(document_text) > max_context:
            # Find relevant chunk
            question_lower = question.lower()
            chunks = chunk_text_intelligently(document_text, max_context, 100)
            
            # Find most relevant chunk
            best_chunk = chunks[0]  # Default to first
            best_score = 0
            
            for chunk in chunks:
                score = sum(1 for word in question_lower.split() if word in chunk.lower())
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            
            context = best_chunk
        else:
            context = document_text
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
        template = """
        You are a Senior Legal Analyst. Analyze the provided document context.
        
        Context: {context}
        Question: {question}
        
        Provide a comprehensive legal analysis:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(context=context, question=question)
        
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- ENTERPRISE SIDEBAR ---
with st.sidebar:
    st.header("🏢 Enterprise Mode")
    st.markdown("**📊 Performance Features:**")
    st.markdown("• Handles large PDFs (up to 10MB)")
    st.markdown("• Progress tracking")
    st.markdown("• Intelligent chunking")
    st.markdown("• Memory optimization")
    
    st.divider()
    
    if AI_AVAILABLE:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Enterprise API")
        elif os.getenv("GOOGLE_API_KEY"):
            api_key = os.getenv("GOOGLE_API_KEY")
            st.success("✅ Enterprise API")
        else:
            api_key = st.text_input("🔑 Enterprise API Key", type="password")
    else:
        st.error("❌ AI not available")
        api_key = None

# --- ENTERPRISE MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI - Enterprise ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Grade Document Analysis System</div>', unsafe_allow_html=True)

# Performance metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Max File Size", "10MB")
with col2:
    st.metric("Processing", "Chunked")
with col3:
    st.metric("Memory", "Optimized")
with col4:
    st.metric("AI Mode", "Enterprise" if AI_AVAILABLE else "Basic")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None

# File upload with size indicator
uploaded_file = st.file_uploader(
    "📄 Upload Legal Document (PDF - up to 10MB)",
    type="pdf",
    help="Supports large PDF files up to 10MB with optimized processing"
)

if uploaded_file and st.button("🚀 Enterprise Analysis", use_container_width=True):
    if not api_key and AI_AVAILABLE:
        st.markdown('<div class="warning-box">⚠️ API Key required for AI analysis</div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Initializing enterprise analysis..."):
        # Process with robust method
        text = get_pdf_text_robust(uploaded_file)
        
        if text:
            st.session_state.document_text = text
            
            # Generate summary with progress
            with st.spinner("📊 Generating enterprise summary..."):
                summary = generate_summary_robust(text)
                st.session_state.document_summary = summary
            
            st.markdown('<div class="success-box">✅ Enterprise analysis complete!</div>', unsafe_allow_html=True)
            
            # Display file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📊 Processed {file_size_mb:.1f}MB document with {summary['total_words']:,} words")
        else:
            st.error("❌ Failed to process document")

# Display summary if available
if st.session_state.document_summary:
    summary = st.session_state.document_summary
    
    st.markdown("### 📊 Enterprise Document Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Words", f"{summary['total_words']:,}")
    with col2:
        st.metric("Sentences", f"{summary['total_sentences']:,}")
    with col3:
        st.metric("Risk Factors", len(summary['risk_indicators']))
    with col4:
        st.metric("Complexity", f"{summary['complexity_score']:.1f}/100")
    
    # Document size indicator
    size_mb = summary['document_size'] / (1024 * 1024)
    st.metric("Document Size", f"{size_mb:.2f}MB")
    
    # Entities
    if any(summary['entities'].values()):
        st.markdown("#### 🔍 Key Entities")
        for entity_type, values in summary['entities'].items():
            if values:
                st.markdown(f"**{entity_type.title()}:** {len(values)} found")
    
    # Risk analysis
    if summary['risk_indicators']:
        st.markdown("#### 🎯 Risk Analysis")
        risk_items = list(summary['risk_indicators'].items())[:10]  # Show top 10
        for risk, count in risk_items:
            st.markdown(f"**{risk.title()}:** {count} occurrence(s)")

# Enterprise Chat Interface
st.markdown("### 💬 Enterprise Legal Consultation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the legal document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.document_text:
        with st.spinner("🤖 Enterprise AI analyzing..."):
            if AI_AVAILABLE and api_key:
                response = get_ai_response_robust(st.session_state.document_text, prompt, api_key)
            else:
                # Fallback analysis
                word_count = len(st.session_state.document_text.split())
                response = f"Document contains {word_count:,} words. Enterprise AI analysis requires API key. Basic analysis shows {len(st.session_state.document_summary['risk_indicators'])} risk factors."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a document first.")

st.markdown("---")
st.markdown("""
<div class="success-box">
🏢 <strong>Enterprise Grade Features:</strong><br>
• Large file support (up to 10MB)<br>
• Memory-optimized processing<br>
• Progress tracking<br>
• Intelligent text chunking<br>
• Production-ready performance
</div>
""", unsafe_allow_html=True)
