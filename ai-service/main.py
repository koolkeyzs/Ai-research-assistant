from contextlib import asyncconextmanager
from fastapi import FastAPI
from rag import extract_text, chunk_text, embed_chunks


app = FastAPI(title="AI Research Assistant - Python Service")


@app.get("/health")
def health():
    return {"status": "ok"}