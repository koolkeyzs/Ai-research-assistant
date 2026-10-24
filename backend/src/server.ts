import express from "express";
import type { Request, Response } from "express";
import cors from "cors";
import multer from "multer";
import "dotenv/config";
import { pool } from "./db.js";

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL ?? "http://127.0.0.1:8000";

const TOP_K = 5;
const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type ProcessResult = {
  filename: string;
  chunk_count: number;
  chunks: string[];
  embeddings: number[][];
};

type ChunkRow = { content: string; embedding: number[] };

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

async function postJson<T>(path: string, body: unknown): Promise<T | null> {
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

app.use(cors());
app.use(express.json());

app.get("/api/health", (req: Request, res: Response) => {
  res.json({ status: "ok" });
});

app.get("/api/health/db", async (req: Request, res: Response) => {
  try {
    const result = await pool.query("SELECT current_database() AS db");
    res.json({ status: "ok", database: result.rows[0].db });
  } catch {
    res.status(500).json({ detail: "Database unavailable." });
  }
});

app.post(
  "/api/documents",
  upload.single("file"),
  async (req: Request, res: Response) => {
    if (!req.file) {
      return res.status(400).json({ detail: "No file uploaded." });
    }

    if (req.file.mimetype !== "application/pdf") {
      return res.status(415).json({ detail: "Only PDF files are accepted." });
    }

    const form = new FormData();
    form.append(
      "file",
      new Blob([new Uint8Array(req.file.buffer)], { type: "application/pdf" }),
      req.file.originalname
    );

    let pyResponse: globalThis.Response;

    try {
      pyResponse = await fetch(`${PYTHON_SERVICE_URL}/process`, {
        method: "POST",
        body: form,
      });
    } catch {
      return res.status(502).json({ detail: "AI service is unavailable." });
    }

    if (!pyResponse.ok) {
      const error = (await pyResponse.json().catch(() => null)) as {
        detail?: string;
      } | null;
      return res
        .status(pyResponse.status)
        .json({ detail: error?.detail ?? "Processing failed." });
    }

    const result = (await pyResponse.json()) as ProcessResult;

    if (result.chunks.length !== result.embeddings.length) {
      return res
        .status(502)
        .json({ detail: "AI service returned inconsistent data." });
    }

    const client = await pool.connect();

    try {
      await client.query("BEGIN");

      const doc = await client.query(
        "INSERT INTO documents (filename, chunk_count) VALUES ($1, $2) RETURNING id",
        [result.filename, result.chunk_count]
      );
      const docId: string = doc.rows[0].id;

      for (let i = 0; i < result.chunks.length; i++) {
        await client.query(
          "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES ($1, $2, $3, $4)",
          [docId, i, result.chunks[i], result.embeddings[i]]
        );
      }

      await client.query("COMMIT");

      res.status(201).json({
        doc_id: docId,
        filename: result.filename,
        chunk_count: result.chunk_count,
      });
    } catch (err) {
      console.error(err);
      await client.query("ROLLBACK");
      res.status(500).json({ detail: "Failed to save the document." });
    } finally {
      client.release();
    }
  }
);

app.post("/api/documents/:id/ask", async (req: Request, res: Response) => {
  const id = req.params.id;
  const question: unknown = req.body?.question;

  if (typeof id !== "string" || !UUID_RE.test(id)) {
    return res.status(404).json({ detail: "Document not found." });
  }

  if (
    typeof question !== "string" ||
    question.trim() === "" ||
    question.length > 2000
  ) {
    return res
      .status(400)
      .json({ detail: "Question must be 1-2000 characters." });
  }

  const doc = await pool.query("SELECT id FROM documents WHERE id = $1", [id]);
  if (doc.rowCount === 0) {
    return res.status(404).json({ detail: "Document not found." });
  }

  const embedded = await postJson<{ embedding: number[] }>("/embed", {
    text: question,
  });
  if (!embedded) {
    return res.status(502).json({ detail: "AI service is unavailable." });
  }

  const rows = await pool.query<ChunkRow>(
    "SELECT content, embedding FROM chunks WHERE document_id = $1",
    [id]
  );

  const top = rows.rows
    .map((row) => ({
      content: row.content,
      score: cosineSimilarity(embedded.embedding, row.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);

  const generated = await postJson<{ answer: string }>("/generate", {
    question,
    chunks: top.map((c) => c.content),
  });
  if (!generated) {
    return res.status(502).json({ detail: "Answer generation failed." });
  }

  res.json({
    answer: generated.answer,
    sources: top.map((c) => ({ text: c.content, score: c.score })),
  });
});

app.listen(PORT, () => {
  console.log(`Listening at port ${PORT}`);
});