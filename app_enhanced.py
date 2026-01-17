import streamlit as st
import os
import re
import PyPDF2

# --- ENHANCED CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI - Enhanced", page_icon="📊", layout="wide")

# --- ENHANCED UI STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800;}
    .sub-header {color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;}
    .success-box {padding: 1rem; background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534;}
    .error-box {padding: 1rem; background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b;}
</style>
""", unsafe_allow_html=True)

# --- ENHANCED IMPORTS (ISOLATED) ---
try:
    import pandas as pd
    import plotly.express as px
    ENHANCED_AVAILABLE = True
    st.success("✅ Enhanced Components Loaded")
except ImportError as e:
    ENHANCED_AVAILABLE = False
    st.error(f"❌ Enhanced Import Error: {e}")

# --- ENHANCED FUNCTIONS ---
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

def extract_entities(text):
    """Extract legal entities"""
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

def create_risk_chart(risk_indicators):
    """Create risk visualization chart"""
    if not ENHANCED_AVAILABLE or not risk_indicators:
        return None
    
    try:
        df = pd.DataFrame(list(risk_indicators.items()), columns=['Risk Factor', 'Count'])
        fig = px.bar(df, x='Risk Factor', y='Count', title='Risk Indicators Analysis',
                      color='Count', color_continuous_scale='Reds')
        return fig
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

# --- ENHANCED SIDEBAR ---
with st.sidebar:
    st.header("📊 Enhanced Features")
    if ENHANCED_AVAILABLE:
        st.success("✅ Enhanced Mode Available")
    else:
        st.error("❌ Enhanced Components Not Available")
    
    st.markdown("**Features:**")
    st.markdown("• Entity extraction")
    st.markdown("• Risk analysis")
    st.markdown("• Visual charts")
    st.markdown("• Document metrics")

# --- ENHANCED MAIN UI ---
st.markdown('<div class="main-header">Legal Insight AI - Enhanced 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Document Analysis System</div>', unsafe_allow_html=True)

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file and st.button("Enhanced Analysis"):
    with st.spinner("Processing..."):
        text = get_pdf_text(uploaded_file)
        if text:
            st.session_state.document_text = text
            st.success("✅ Document processed successfully!")
            
            # Perform enhanced analysis
            entities = extract_entities(text)
            risk_analysis = analyze_risk(text)
            
            # Display results
            st.markdown("### 📊 Document Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Characters", len(text))
                st.metric("Words", len(text.split()))
            with col2:
                st.metric("Risk Factors", len(risk_analysis))
                st.metric("Entities Found", sum(len(v) for v in entities.values()))
            
            # Display entities
            if any(entities.values()):
                st.markdown("#### 🔍 Key Entities")
                for entity_type, values in entities.items():
                    if values:
                        st.markdown(f"**{entity_type.title()}:** {', '.join(values[:3])}")
            
            # Display risk analysis
            if risk_analysis:
                st.markdown("#### 🎯 Risk Analysis")
                fig = create_risk_chart(risk_analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Risk Details"):
                    for risk, count in risk_analysis.items():
                        st.markdown(f"**{risk.title()}:** {count} occurrence(s)")
            else:
                st.info("🎉 No significant risk indicators detected.")
        else:
            st.error("❌ Unable to process document")

st.markdown('<div class="success-box">✅ Enhanced Mode: Advanced analysis with visualizations</div>', unsafe_allow_html=True)
