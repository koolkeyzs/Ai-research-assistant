import tempfile
from pathlib import Path

import streamlit as st

from rag import extract_text, chunk_text, embed_chunks, ask

st.set_page_config(page_title="AI Research Assistant", page_icon="📄")
st.title("AI Research Assistant")
st.caption("Upload a PDF and ask questions. Answers come only from the document.")


@st.cache_resource(show_spinner="Reading and embedding the PDF...")
def process_pdf(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        text = extract_text(tmp_path)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
    finally:
        tmp_path.unlink(missing_ok=True)
    return chunks, embeddings


uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

try:
    chunks, embeddings = process_pdf(uploaded.getvalue())
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"{uploaded.name}: {len(chunks)} chunks ready")

question = st.text_input("Ask a question about this document").strip()

if question:
    with st.spinner("Thinking..."):
        try:
            answer, sources = ask(question, chunks, embeddings, top_k=5)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.caption(f"similarity {s['score']:.2f}")
            st.text(s["chunk"])