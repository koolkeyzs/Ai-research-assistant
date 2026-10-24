import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from rag import (
    extract_text,
    chunk_text,
    embed_chunks,
    embed_query,
    generate_answer,
)



# 1. APPLICATION CONFIGURATION


MAX_FILE_MB = 20

app = FastAPI(
    title="AI Research Assistant - Python Service"
)



# 2. REQUEST MODELS


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    chunks: list[str] = Field(min_length=1, max_length=20)



# 3. HEALTH CHECK


@app.get("/health")
def health():
    return {"status": "ok"}



# 4. PROCESS AN UPLOADED PDF


@app.post("/process")
def process(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are accepted."
        )

    data = file.file.read()

    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File larger than {MAX_FILE_MB} MB."
        )

    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="File is not a valid PDF."
        )

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        text = extract_text(tmp_path)

        chunks = chunk_text(text)

        embeddings = embed_chunks(chunks)

        return {
            "filename": file.filename,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "embeddings": embeddings.tolist()
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF could not be processed."
        ) from exc

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

        file.file.close()



# 5. EMBED A QUESTION OR OTHER TEXT


@app.post("/embed")
def embed(request: EmbedRequest):
    try:
        embedding = embed_query(request.text)

        return {
            "embedding": embedding.tolist()
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc



# 6. GENERATE AN ANSWER USING GEMINI


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        answer = generate_answer(
            request.question,
            request.chunks
        )

        return {
            "answer": answer
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail="AI answer generation failed."
        ) from exc