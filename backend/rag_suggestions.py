"""
RAG-powered AI suggestions for VitalFlow AI.

Pipeline Overview
-----------------
1. BUILD PHASE (once, at startup or on first call):
   - Load medical_guidelines.txt from the knowledge_base directory.
   - Split the text into chunks delimited by "---CHUNK---".
   - Embed each chunk using sentence-transformers (all-MiniLM-L6-v2, ~80 MB).
   - Store embeddings in a FAISS IndexFlatIP (inner-product / cosine similarity
     after L2-normalisation) with integer IDs mapping back to chunk text.

2. QUERY PHASE (each API call):
   - Construct a short clinical query string from the patient's disease type,
     risk level, and vital readings.
   - Embed the query with the same sentence-transformers model.
   - Retrieve the top-K most relevant chunks from FAISS (K=3 by default).
   - Inject the retrieved chunks as a "Clinical Context" block into the Gemini
     prompt, instructing the model to ground its answer in that evidence.
   - Return the model's response as a structured suggestion.

Why FAISS over a hosted vector DB?
    FAISS runs entirely in-process (no additional service to start), making it
    ideal for a student project on a local machine.  The index is rebuilt at
    startup (< 1 second for ~10 chunks) so no persistent vector store is needed.
    Swap `build_faiss_index` for a ChromaDB or Pinecone client to scale to
    thousands of documents without changing the retrieval interface.
"""

import os
import logging
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Optional imports — graceful fallback if packages not installed ─────────────
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logging.warning(
        "[RAG] faiss-cpu not installed. RAG suggestions will fall back to "
        "basic prompt without retrieved context.  Run: pip install faiss-cpu"
    )

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logging.warning(
        "[RAG] sentence-transformers not installed. RAG disabled. "
        "Run: pip install sentence-transformers"
    )

from google import genai

# ── Configuration ──────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(__file__), "knowledge_base", "medical_guidelines.txt"
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
TOP_K           = 3                      # number of chunks retrieved per query
GEMINI_MODEL    = "gemini-2.0-flash"

# Module-level cache
_embedder:    Optional["SentenceTransformer"] = None
_faiss_index  = None
_chunks:      list[str]                       = []

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Sanitisation ───────────────────────────────────────────────────────────────

def _sanitize(value: str, max_length: int = 100) -> str:
    """Strip prompt-injection characters and truncate user-supplied strings."""
    return value.replace("{", "").replace("}", "").replace("\n", " ").strip()[:max_length]


# ── Knowledge Base Loading ─────────────────────────────────────────────────────

def _load_chunks() -> list[str]:
    """
    Read the knowledge base file and split into chunks.
    Returns a list of non-empty chunk strings.
    """
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        logging.error("[RAG] Knowledge base not found at %s", KNOWLEDGE_BASE_PATH)
        return []

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as fh:
        raw = fh.read()

    chunks = [
        c.strip()
        for c in raw.split("---CHUNK---")
        if c.strip() and not c.strip().startswith("#")
    ]
    logging.info("[RAG] Loaded %d chunks from knowledge base.", len(chunks))
    return chunks


# ── FAISS Index Construction ───────────────────────────────────────────────────

def build_faiss_index() -> bool:
    """
    Embed all knowledge-base chunks and build a FAISS index.
    Populates module-level _embedder, _faiss_index, _chunks.
    Returns True on success, False if dependencies are missing.
    """
    global _embedder, _faiss_index, _chunks

    if not (_FAISS_AVAILABLE and _ST_AVAILABLE):
        return False

    if _faiss_index is not None:
        return True   # already built

    _chunks = _load_chunks()
    if not _chunks:
        return False

    _embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Embed → normalise → add to FAISS inner-product index
    # After L2 normalisation, inner product == cosine similarity
    embeddings = _embedder.encode(_chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)                 # cosine similarity
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, np.arange(len(_chunks), dtype=np.int64))

    _faiss_index = index
    logging.info("[RAG] FAISS index built: %d vectors, dim=%d.", len(_chunks), dim)
    return True


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = TOP_K) -> str:
    """
    Embed *query* and return the top-k most relevant knowledge-base chunks
    concatenated as a single context block.
    Returns an empty string if the index is unavailable.
    """
    if _faiss_index is None or _embedder is None:
        return ""

    q_emb = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, ids = _faiss_index.search(q_emb, k)

    retrieved = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(_chunks):
            continue
        retrieved.append(f"[score={score:.2f}]\n{_chunks[idx]}")

    return "\n\n".join(retrieved)


# ── Prompt Assembly ───────────────────────────────────────────────────────────

def _build_readings_text(latest_readings: dict) -> str:
    lines = []
    if latest_readings.get("blood_sugar") is not None:
        lines.append(f"Blood Sugar: {latest_readings['blood_sugar']} mg/dL")
    if latest_readings.get("systolic_bp") is not None:
        diastolic = latest_readings.get("diastolic_bp", "?")
        lines.append(f"Blood Pressure: {latest_readings['systolic_bp']}/{diastolic} mmHg")
    if latest_readings.get("heart_rate") is not None:
        lines.append(f"Heart Rate: {latest_readings['heart_rate']} bpm")
    if latest_readings.get("temperature") is not None:
        lines.append(f"Temperature: {latest_readings['temperature']}°C")
    if latest_readings.get("spo2") is not None:
        lines.append(f"SpO2: {latest_readings['spo2']}%")
    return "\n".join(lines) if lines else "No recent readings available."


# ── Public API ────────────────────────────────────────────────────────────────

def get_rag_suggestions(
    patient_name:    str,
    disease_type:    str,
    age:             int,
    risk_level:      str,
    latest_readings: dict,
) -> str:
    """
    Generate clinically grounded health suggestions using a RAG pipeline.

    The function:
      1. Ensures the FAISS index is built (cached after the first call).
      2. Constructs a retrieval query from the patient's clinical profile.
      3. Fetches the top-3 relevant medical guideline chunks.
      4. Injects those chunks into a structured Gemini prompt.
      5. Returns the model's response text.

    Falls back to a basic prompt (without retrieved context) if faiss-cpu or
    sentence-transformers are not installed.
    """
    # Ensure the index is ready (builds once, then uses cache)
    rag_enabled = build_faiss_index()

    safe_name    = _sanitize(patient_name)
    safe_disease = _sanitize(disease_type)
    safe_risk    = _sanitize(risk_level, max_length=20)
    readings_text = _build_readings_text(latest_readings)

    # Build the retrieval query — combine condition + risk + most abnormal vital
    query_parts = [safe_disease, f"{safe_risk} risk", "recovery management"]
    if latest_readings.get("blood_sugar", 0) and latest_readings["blood_sugar"] > 180:
        query_parts.append("hyperglycemia")
    if latest_readings.get("systolic_bp", 0) and latest_readings["systolic_bp"] > 140:
        query_parts.append("hypertension blood pressure target")
    if latest_readings.get("spo2", 100) and latest_readings["spo2"] < 94:
        query_parts.append("low oxygen saturation SpO2")
    if latest_readings.get("heart_rate", 0) and latest_readings["heart_rate"] > 100:
        query_parts.append("tachycardia post-operative")
    retrieval_query = " ".join(query_parts)

    # Retrieve relevant guideline chunks
    clinical_context = retrieve_context(retrieval_query) if rag_enabled else ""

    # Build the prompt
    context_block = (
        f"""
---CLINICAL GUIDELINES (retrieved from evidence-based sources) ---
{clinical_context}
---END GUIDELINES---
"""
        if clinical_context
        else ""
    )

    prompt = f"""You are a clinical decision-support assistant for VitalFlow AI, a patient recovery monitoring system.
Your suggestions must be grounded in the clinical guidelines provided below.
Do NOT make up drug names, dosages, or medical facts not present in the context.
Do NOT provide a diagnosis. Remind the patient to consult their physician for any concerns.
{context_block}
Patient Profile:
- Name: {safe_name}
- Age: {age}
- Condition: {safe_disease}
- Current Risk Level: {safe_risk}

Latest Vitals:
{readings_text}

Based on the above clinical guidelines and patient data, provide:
1. **Food Recommendations** — 5 specific foods that support recovery for this condition (with brief reason), and 3 foods to avoid.
2. **Lifestyle Tips** — 3 practical daily habits tailored to this patient's risk level and current vitals.
3. **Warning Signs** — 2 specific symptoms this patient should watch for given their current readings and condition, and what to do if they occur.

Format clearly with the three sections. Be concise and practical."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = get_rag_suggestions(
        patient_name="Test Patient",
        disease_type="diabetes",
        age=58,
        risk_level="High",
        latest_readings={
            "blood_sugar": 210,
            "systolic_bp": 145,
            "diastolic_bp": 92,
            "heart_rate": 96,
            "spo2": 95,
        },
    )
    print(result)
