import json
import time
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.vectorstore.chroma_manager import chroma_manager
from app.llm.ollama_client import ollama_client
from app.models.response_models import SourceChunk
from app.models.db_models import ChatHistory, User
from app.utils.logger import logger, log_execution_time
from app.utils.exceptions import VectorStoreError, LLMQueryError

class RAGService:
    """Service orchestrating the end-to-end Retrieval Augmented Generation (RAG) pipeline."""

    def __init__(self):
        self.prompt_template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise and well-explained.

Context:
{context}

Question: {question}

Helpful Answer:"""

    def query_knowledge_base(
        self,
        question: str,
        k: int = 4,
        model_override: Optional[str] = None,
        db: Optional[Session] = None,
        user: Optional[User] = None
    ) -> Dict[str, Any]:
        """Executes vector cosine similarity retrieval and Ollama LLM response synthesis."""
        
        # Verify vector store has indexed documents
        doc_count = chroma_manager.get_document_count()
        if doc_count == 0:
            raise VectorStoreError("Vector database is empty. Please upload at least one PDF document before querying.")

        retriever = chroma_manager.get_retriever(k=k)
        llm = ollama_client.get_llm(model_override=model_override)

        PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        logger.info(f"Executing RAG query: '{question}'")
        start_time = time.time()
        
        try:
            with log_execution_time(f"Ollama RAG Query [{model_override or ollama_client.model}]"):
                result = qa_chain.invoke({"query": question})
        except Exception as e:
            logger.error(f"Error executing RAG query chain: {str(e)}", exc_info=True)
            raise LLMQueryError(f"LLM query execution failed: {str(e)}")

        elapsed_ms = (time.time() - start_time) * 1000

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        formatted_sources = [
            SourceChunk(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in source_docs
        ]

        sources_json = json.dumps([{"content": s.page_content, "metadata": s.metadata} for s in formatted_sources])

        # Record in SQLite chat history if db session provided
        if db:
            try:
                chat_record = ChatHistory(
                    user_id=user.id if user else None,
                    question=question,
                    answer=answer,
                    source_docs_json=sources_json,
                    response_time_ms=elapsed_ms
                )
                db.add(chat_record)
                db.commit()
            except Exception as dbe:
                logger.error(f"Failed to record chat history into SQLite: {str(dbe)}")

        return {
            "query": question,
            "answer": answer,
            "source_documents": formatted_sources,
            "response_time_ms": round(elapsed_ms, 2)
        }

# Singleton instance
rag_service = RAGService()
