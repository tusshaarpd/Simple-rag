import os
import tempfile

import pandas as pd
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.title("Chat with your Documents")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored or logged.",
    )
    st.markdown("---")
    st.markdown(
        "**Supported files:** PDF, Excel (.xlsx)  \n"
        "**Max file size:** 10 MB"
    )

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ── Text extraction helpers ───────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    finally:
        os.unlink(tmp_path)


def extract_text_from_excel(uploaded_file) -> str:
    """Convert each row of an Excel file to a readable string."""
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    # Represent each row as "col1: val1 | col2: val2 | ..."
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
        rows.append(row_text)
    return "\n".join(rows)


# ── Vector store builder ──────────────────────────────────────────────────────
def build_vectorstore(text: str, api_key: str) -> FAISS:
    """Chunk text and create a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)


# ── Main UI ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PDF or Excel file",
    type=["pdf", "xlsx"],
)

if uploaded_file is not None:
    # Enforce file size limit
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        st.error(f"File is too large ({len(file_bytes) / 1024 / 1024:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    # Only (re-)process when a new file is uploaded
    if st.session_state.get("processed_file") != uploaded_file.name:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar before uploading.")
            st.stop()

        with st.spinner("Extracting text and building vector store..."):
            try:
                if uploaded_file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = extract_text_from_excel(uploaded_file)

                if not text.strip():
                    st.error("Could not extract any text from the file. Please check the file contents.")
                    st.stop()

                vectorstore = build_vectorstore(text, openai_api_key)
                st.session_state["vectorstore"] = vectorstore
                st.session_state["processed_file"] = uploaded_file.name

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        st.success(f"Ready! '{uploaded_file.name}' has been processed. Ask your question below.")
    else:
        st.info(f"Using already-processed file: **{uploaded_file.name}**")

# ── Question & Answer ─────────────────────────────────────────────────────────
if "vectorstore" in st.session_state:
    question = st.text_input("Ask a question about your document:")

    if question:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            st.stop()

        with st.spinner("Thinking..."):
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=openai_api_key,
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state["vectorstore"].as_retriever(
                        search_kwargs={"k": 4}
                    ),
                )
                result = qa_chain.invoke({"query": question})
                answer = result.get("result", "No answer found.")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

        st.markdown("### Answer")
        st.write(answer)
else:
    st.info("Upload a document above to get started.")
