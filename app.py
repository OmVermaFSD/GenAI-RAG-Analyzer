streamlit as st
import os
import 
st.set_page_config(
    page_title="et An",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {ont-size: 2.5rem; color: #1f2; margin-ottom: 1rem;}
    .sub-text {color: #6b78;font-size: 1.1re; m}
    .dislimer {ont-size: 0.8rem; color: #9ca3af; margin-top: re; padding-top: 1e; border: 1px solid #ee;}
</style>
""",unsfe_allow_html=True)

def get_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.page:
           txt+= page.extract_text() or ""
    except Exception a e:
        s return None
    return text

def speet_ste, questionl = tleenertie(el"geminipro teemraure, olaieaie
    
     ei omt    temlat
       ae  enor  ayt n terovie ntratdocument contet tcl.
        es:
     e  specificaes r etins if aille
     ilit tetl rss in o
      t ases t he document state: "Not found in th oet."
     inain n esionloe
        ntext ontext
    uetion usetion    
    ls:
          rotematlaeee intiesonte =teloit(
        lm cainpestf reteeectosoaseter,
        es:
         returnhirun(estient."

with st.sidbar:
    stitle(ues")
    t.info("Sil oental\no P e eui")
    s.rets"
    st.rdo"-  e etn")
    akon("-Ks")
    t.("-entsa")
    st.n("-  eteas")

st.markdown('<div class="main-header">Document Analy</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload  ments for instt anass and e extraction.</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "et_te" not in st.session_state:
    st.session_state.et_te = 

uploaded_file = st.file_uploader("Upload PF Dment", type="pdf")

if uploaded_file and st.button("Analyze Document"):
    with st.spinner("cing document.."):
        text = get_pdf_text(uploaded_file)
        if text:
            t.sessin_satecuent_texttext
            st.sessont roessedes)
            st.("Document nex ccesll")
       else:
            st.error("Unable to redocument.  nsure it is a ad .")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.cat_message("user"):
       st.markown(prompt)
    
    if st.session_state.et_tet:
        with s.spinner("nling..."):
            response = et_sis(st.sesson_state.et_tet, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a document first.")

s.markdown('<div class="disclaimer">This is a si doent ntt proesoa g advce  alsi ae cous.</div>', unsafe_allow_html=True)
