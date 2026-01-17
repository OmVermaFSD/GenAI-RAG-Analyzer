import streamlit as st
import os
import PyPDF2
import google.generativeai as genai

# Custom RAG Implementation - No LangChain Dependencies
class CustomTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

class CustomVectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add_texts(self, texts):
        self.chunks.extend(texts)
    
    def similarity_search(self, query, k=3):
        # Simple keyword matching as similarity
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, i, chunk))
        
        # Sort by score and return top k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, _, chunk in scored_chunks[:k]]

class CustomPromptTemplate:
    def __init__(self, template):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)

class CustomRAGChain:
    def __init__(self, llm, vector_store, prompt):
        self.llm = llm
        self.vector_store = vector_store
        self.prompt = prompt
    
    def invoke(self, query):
        # Get relevant chunks
        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n\n".join(docs)
        
        # Format prompt
        formatted_prompt = self.prompt.format(context=context, question=query)
        
        # Get response
        response = self.llm.generate_content(formatted_prompt)
        return response.content

st.set_page_config(page_title="GenAI RAG Analyst", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A;}
    .status-ok {background-color: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 12px; font-weight: bold;}
    .status-err {background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 12px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

def get_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        return None
    return text

def get_expert_response(vector_store, question, api_key):
    try:
        genai.configure(api_key=api_key)
        # Try different model names that might be available
        try:
            llm = genai.GenerativeModel('models/gemini-1.0-pro')
        except:
            try:
                llm = genai.GenerativeModel('gemini-1.0-pro')
            except:
                try:
                    llm = genai.GenerativeModel('models/gemini-pro-latest')
                except:
                    llm = genai.GenerativeModel('gemini-pro-latest')
        
        template = """
        You are an Expert Strategy Consultant at Deloitte. Answer based ONLY on the Context below.
        Use professional business language and bullet points.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = CustomPromptTemplate(template)
        chain = CustomRAGChain(llm, vector_store, prompt)
        
        return chain.invoke(question)
        
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.markdown('<span class="status-ok">✅ API Key Loaded</span>', unsafe_allow_html=True)
    elif os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.markdown('<span class="status-ok">✅ API Key Loaded</span>', unsafe_allow_html=True)
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if not api_key: st.markdown('<span class="status-err">🔴 Waiting for Key</span>', unsafe_allow_html=True)

# Main UI
st.markdown('<div class="main-header">GenAI Document Analyst</div>', unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

uploaded_file = st.file_uploader("Upload PDF Report", type="pdf")

if uploaded_file and st.button("🚀 Process Document"):
    if not api_key:
        st.error("❌ API Key Missing.")
    else:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_file)
            if raw_text:
                # Create vector store
                vector_store = CustomVectorStore()
                
                # Split text
                splitter = CustomTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(raw_text)
                
                # Add to vector store
                vector_store.add_texts(chunks)
                
                # Store in session
                st.session_state.vector_store = vector_store
                st.success("✅ Analysis Ready!")
                st.info(f"📄 Document processed into {len(chunks)} chunks")
            else:
                st.error("❌ Document Empty or Scanned.")

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if st.session_state.vector_store:
            with st.spinner("Analyzing..."):
                response = get_expert_response(st.session_state.vector_store, prompt, api_key)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Please process a document first.")
