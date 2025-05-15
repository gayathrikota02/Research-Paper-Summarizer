import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TORCH_HOME"] = "./.torch"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import time
import logging
import streamlit as st
from langchain.llms.base import LLM
from typing import Optional, List
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import SequentialChain, TransformChain
from pydantic import PrivateAttr
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import TimeoutError as FuturesTimeoutError


# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== Gemini Wrapper ====
class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash-lite"
    _model: any = PrivateAttr()
    timeout: int = 60  # seconds

    def __init__(self, model_name="gemini-2.0-flash-lite", **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # put your real env var name here
        self._model = genai.GenerativeModel(model_name)

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._model.generate_content, prompt)
            try:
                response = future.result(timeout=self.timeout)
                return response.text
            except FuturesTimeoutError:
                return "Error: LLM call timed out."


# ==== Utility Functions ====
def load_and_split_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vectorstore(docs):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding_model)

def build_chains(gemini_llm, docs, vectorstore):
    summarize_chain = load_summarize_chain(gemini_llm, chain_type="refine")

    def recommend_fn(inputs):
        summary = inputs["output_text"]
        similar_docs = vectorstore.similarity_search(summary, k=3)
        return {"recommendations": "\n".join([doc.page_content[:500] for doc in similar_docs])}

    recommendation_chain = TransformChain(
        input_variables=["output_text"],
        output_variables=["recommendations"],
        transform=recommend_fn
    )

    return SequentialChain(
        chains=[summarize_chain, recommendation_chain],
        input_variables=["input_documents"],
        output_variables=["output_text", "recommendations"],
        verbose=False
    )

def answer_question(gemini_llm, vectorstore, question: str, k: int = 3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    return gemini_llm(prompt)

# New: External paper recommendations via Gemini
def recommend_external_papers_with_context(gemini_llm, summary, vectorstore, top_k=3, num_papers=5):
    similar_docs = vectorstore.similarity_search(summary, k=top_k)
    similar_content = "\n\n".join([doc.page_content for doc in similar_docs])

    prompt = f"""
You are a highly knowledgeable research assistant.

Here is a research paper summary:
\"\"\"
{summary}
\"\"\"

And here are some related documents from a local research paper collection:
\"\"\"
{similar_content}
\"\"\"

Based on both, recommend {num_papers} other external research papers, academic articles, or notable works that would be relevant to the main topic and concepts discussed. These should NOT be from the provided content but should be well-known or significant works in this field.

For each, include:
- Title or Topic
- A brief description (1-2 lines)
- (Optional) publication year if known

List of related external research papers:
"""
    response = gemini_llm(prompt)
    return response


# ==== Streamlit App ====
st.set_page_config(page_title="PDF Summarizer & QA", layout="wide")
st.title("📄 Research Paper Summarizer and Recommendation Engine")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file to disk
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        with st.spinner("Processing and summarizing PDF..."):
            docs = load_and_split_pdf(temp_path)
            vectorstore = create_vectorstore(docs)
            gemini_llm = GeminiLLM()
            workflow = build_chains(gemini_llm, docs, vectorstore)

            result = workflow.invoke({"input_documents": docs})
            summary = result.get("output_text", "No summary generated.")
            recommendations = result.get("recommendations", "No recommendations generated.")

            st.session_state.last_file = uploaded_file.name
            st.session_state.docs = docs
            st.session_state.vectorstore = vectorstore
            st.session_state.summary = summary
            st.session_state.recommendations = recommendations
            st.session_state.llm = gemini_llm
    else:
        docs = st.session_state.docs
        vectorstore = st.session_state.vectorstore
        summary = st.session_state.summary
        recommendations = st.session_state.recommendations
        gemini_llm = st.session_state.llm

    st.subheader("📌 Summary")
    st.write(summary)

    st.subheader("👯‍♀️ Similarity Test with summary")
    st.write(recommendations)


    st.subheader("❓ Ask Questions")
    question_input = st.text_input("Type your question here:")
    if question_input:
        with st.spinner("Generating answer..."):
            answer = answer_question(gemini_llm, vectorstore, question_input)
            st.success(answer)

    st.subheader("📖 External Research Paper Recommendations (with Context)")
    top_k = st.slider("Select number of local similar papers to use as context", 1, 10, 3)
    num_papers = st.slider("How many external recommendations?", 1, 10, 5)

    if st.button("🔍 Suggest Related Research Papers"):
        with st.spinner("Consulting Gemini with context for related research papers..."):
            recommendations_response = recommend_external_papers_with_context(
                gemini_llm, summary, vectorstore, top_k=top_k, num_papers=num_papers
            )
            st.write(recommendations_response)
