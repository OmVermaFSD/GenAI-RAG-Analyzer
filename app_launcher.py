import streamlit as st
import os

# --- LAUNCHER CONFIGURATION ---
st.set_page_config(page_title="Legal Insight AI - Launcher", page_icon="🚀", layout="center")

# --- LAUNCHER STYLING ---
st.markdown("""
<style>
    .launcher-header {font-size: 3rem; color: #1e293b; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; text-align: center;}
    .launcher-sub {color: #64748b; font-size: 1.2rem; text-align: center; margin-bottom: 2rem;}
    .mode-card {padding: 2rem; border: 2px solid #e2e8f0; border-radius: 10px; margin: 1rem 0; transition: all 0.3s;}
    .mode-card:hover {border-color: #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .mode-title {font-size: 1.5rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;}
    .mode-desc {color: #64748b; margin-bottom: 1rem;}
    .mode-features {color: #059669; font-size: 0.9rem;}
    .status-working {color: #059669; font-weight: 600;}
    .status-limited {color: #d97706; font-weight: 600;}
    .status-error {color: #dc2626; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# --- LAUNCHER MAIN UI ---
st.markdown('<div class="launcher-header">🚀 Legal Insight AI</div>', unsafe_allow_html=True)
st.markdown('<div class="launcher-sub">Choose Your Analysis Mode</div>', unsafe_allow_html=True)

st.markdown("---")

# Check available modes
basic_available = os.path.exists("app_basic.py")
ai_available = os.path.exists("app_ai.py")
enhanced_available = os.path.exists("app_enhanced.py")

# Basic Mode
if basic_available:
    st.markdown("""
    <div class="mode-card">
        <div class="mode-title">⚖️ Basic Mode</div>
        <div class="mode-desc">Simple document analysis without external dependencies</div>
        <div class="mode-features">
        ✅ Always works<br>
        ✅ No API keys required<br>
        ✅ PDF text extraction<br>
        ✅ Keyword analysis<br>
        ✅ Document summary
        </div>
        <div class="status-working">🟢 Status: Working</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Basic Mode", key="basic", use_container_width=True):
        st.switch_page("app_basic.py")

# AI Mode
if ai_available:
    st.markdown("""
    <div class="mode-card">
        <div class="mode-title">🤖 AI Mode</div>
        <div class="mode-desc">Advanced analysis with Gemini AI</div>
        <div class="mode-features">
        ⚠️ Requires API key<br>
        ⚠️ Needs working imports<br>
        ✅ AI-powered analysis<br>
        ✅ Legal expert responses<br>
        ✅ Contextual understanding
        </div>
        <div class="status-limited">🟡 Status: Dependencies Required</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch AI Mode", key="ai", use_container_width=True):
        st.switch_page("app_ai.py")

# Enhanced Mode
if enhanced_available:
    st.markdown("""
    <div class="mode-card">
        <div class="mode-title">📊 Enhanced Mode</div>
        <div class="mode-desc">Advanced analysis with visualizations</div>
        <div class="mode-features">
        ⚠️ Requires additional packages<br>
        ✅ Entity extraction<br>
        ✅ Risk analysis<br>
        ✅ Visual charts<br>
        ✅ Document metrics
        </div>
        <div class="status-limited">🟡 Status: Optional Dependencies</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Enhanced Mode", key="enhanced", use_container_width=True):
        st.switch_page("app_enhanced.py")

st.markdown("---")

# Troubleshooting Section
with st.expander("🔧 Troubleshooting Guide"):
    st.markdown("""
    ### If Basic Mode Fails:
    - Check if `pypdf2` is installed
    - Verify PDF file is not corrupted
    
    ### If AI Mode Fails:
    - Check `requirements_ai.txt` dependencies
    - Verify Google API key is valid
    - Check internet connection
    
    ### If Enhanced Mode Fails:
    - Install `requirements_enhanced.txt` dependencies
    - Check `pandas` and `plotly` installation
    
    ### Isolation Strategy:
    - Each mode runs independently
    - Errors in one mode don't affect others
    - Start with Basic Mode, then try others
    """)

st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 2rem;">
    🛡️ Fault-Tolerant Design • 🔄 Independent Modes • 🚀 Progressive Enhancement
</div>
""", unsafe_allow_html=True)
