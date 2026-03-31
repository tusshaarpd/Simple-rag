import os
import tempfile

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ── Page config ───────────────────────────────────────────────────────────────
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

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.unlink(tmp_path)


def extract_text_from_excel(uploaded_file) -> str:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    rows = [" | ".join(f"{col}: {val}" for col, val in row.items())
            for _, row in df.iterrows()]
    return "\n".join(rows)


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embeddings(texts: list, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([item.embedding for item in response.data], dtype="float32")


# ── Vector index ──────────────────────────────────────────────────────────────
def build_index(chunks: list, api_key: str):
    embeddings = get_embeddings(chunks, api_key)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


# ── Answer generation ─────────────────────────────────────────────────────────
def answer_question(question: str, index, chunks: list, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    q_embedding = get_embeddings([question], api_key)
    _, indices = index.search(q_embedding, k=4)
    context = "\n\n".join(chunks[i] for i in indices[0] if i < len(chunks))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the context provided. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


# ── Main UI ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

if uploaded_file is not None:
    if len(uploaded_file.getvalue()) > MAX_FILE_SIZE_BYTES:
        st.error(
            f"File is too large "
            f"({len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is 10 MB."
        )
        st.stop()

    if st.session_state.get("processed_file") != uploaded_file.name:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar before uploading.")
            st.stop()

        with st.spinner("Extracting text and building index..."):
            try:
                if uploaded_file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = extract_text_from_excel(uploaded_file)

                if not text.strip():
                    st.error("Could not extract any text from the file.")
                    st.stop()

                chunks = chunk_text(text)
                index, chunks = build_index(chunks, openai_api_key)
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["processed_file"] = uploaded_file.name

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        st.success(f"Ready! '{uploaded_file.name}' processed. Ask your question below.")
    else:
        st.info(f"Using already-processed file: **{uploaded_file.name}**")

# ── Q&A ───────────────────────────────────────────────────────────────────────
if "index" in st.session_state:
    question = st.text_input("Ask a question about your document:")

    if question:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            st.stop()

        with st.spinner("Thinking..."):
            try:
                answer = answer_question(
                    question,
                    st.session_state["index"],
                    st.session_state["chunks"],
                    openai_api_key,
                )
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

        st.markdown("### Answer")
        st.write(answer)
else:
    st.info("Upload a document above to get started.")
