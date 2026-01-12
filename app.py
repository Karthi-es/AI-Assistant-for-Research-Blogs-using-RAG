import logging
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from assistant import Assistant
from gui import AssistantGUI
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE

VECTORSTORE_PATH = "./data/vectorstore"
logger = logging.getLogger(__name__)


def parse_blog_urls(raw_urls: str) -> List[str]:
    if not raw_urls:
        return []

    return [line.strip() for line in raw_urls.splitlines() if line.strip()]


def load_blog_documents(urls: List[str]) -> Tuple[List, List[str]]:
    documents = []
    loaded_sources = []

    requests_kwargs = {
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    }

    for url in urls:
        try:
            loader = WebBaseLoader(url, requests_kwargs=requests_kwargs)
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", url)
            documents.extend(docs)
            loaded_sources.append(url)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load blog URL %s: %s", url, exc)
            st.warning(f"Unable to load {url}. Reason: {exc}")

    return documents, loaded_sources


def load_pdf_documents(uploaded_files) -> Tuple[List, List[str]]:
    documents = []
    loaded_sources = []

    if not uploaded_files:
        return documents, loaded_sources

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", uploaded_file.name)
            documents.extend(docs)
            loaded_sources.append(uploaded_file.name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to process PDF %s: %s", uploaded_file.name, exc)
            st.error(f"Unable to read {uploaded_file.name}. Ensure it is a valid PDF.")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return documents, loaded_sources


def reset_vector_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(path):
        shutil.rmtree(path)


def build_vector_store(blog_urls: List[str], pdf_files) -> Tuple[Chroma | None, List[str]]:
    documents = []
    sources = []

    blog_docs, blog_sources = load_blog_documents(blog_urls)
    documents.extend(blog_docs)
    sources.extend(blog_sources)

    pdf_docs, pdf_sources = load_pdf_documents(pdf_files)
    documents.extend(pdf_docs)
    sources.extend(pdf_sources)

    if not documents:
        st.warning("Add at least one blog URL or PDF before building the knowledge base.")
        return None, []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    reset_vector_directory(VECTORSTORE_PATH)
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=VECTORSTORE_PATH,
    )

    return vector_store, sources


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Research Blog Copilot", page_icon="🧪", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "sources" not in st.session_state:
        st.session_state.sources = []

    st.title("Research Blog Copilot")
    st.caption("Ground answers in the research content you provide via blog links and PDF uploads.")

    with st.sidebar:
        st.header("Knowledge Sources")
        blog_url_input = st.text_area(
            "Research blog URLs",
            placeholder="https://example.com/blog/post-1\nhttps://example.com/blog/post-2",
        )
        uploaded_pdfs = st.file_uploader(
            "Upload research papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Build Knowledge Base", use_container_width=True):
            with st.spinner("Indexing sources..."):
                urls = parse_blog_urls(blog_url_input)
                vector_store, sources = build_vector_store(urls, uploaded_pdfs)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.sources = sources
                st.success(f"Knowledge base built from {len(sources)} sources.")

        if st.button("Reset Conversation", use_container_width=True):
            st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]
            st.info("Conversation reset.")

        if st.session_state.sources:
            st.divider()
            st.caption("Active sources")
            for source in st.session_state.sources:
                st.caption(f"- {source}")

    if not st.session_state.vector_store:
        st.info("Add blog URLs or PDFs, then build the knowledge base to start chatting.")

    llm = ChatGroq(model="llama-3.1-8b-instant")

    assistant = Assistant(
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        message_history=st.session_state.messages,
        vector_store=st.session_state.vector_store,
    )

    gui = AssistantGUI(assistant)
    gui.render()