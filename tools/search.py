"""
tools/search.py
Handles:
  - Tavily web search
  - Qdrant vector DB (RAG on historical events CSV)
  - Robust JSON parser used by all agents
"""

import os
import json
import re
import hashlib
import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ─────────────────────────────────────────────
# Tavily
# ─────────────────────────────────────────────
_tavily = TavilyClient(api_key="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")


def web_search(query: str, max_results: int = 5) -> str:
    """Run a Tavily search and return formatted text context."""
    try:
        results = _tavily.search(query=query, max_results=max_results)
        lines = []
        for r in results.get("results", []):
            lines.append(f"[{r['title']}]\n{r['content'][:300]}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


# ─────────────────────────────────────────────
# Embedder — uses qdrant's built-in fastembed
# or falls back to a tiny manual approach
# ─────────────────────────────────────────────
COLLECTION = "conference_events"
VECTOR_DIM  = 384

_qdrant: QdrantClient | None = None
_model = None


def _get_model():
    """Load sentence-transformers with all warnings suppressed."""
    global _model
    if _model is None:
        import warnings, logging
        # silence every noisy logger sentence-transformers triggers
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

        # suppress the BertModel LOAD REPORT patch
        import transformers.modeling_utils as _mu
        _orig_load = _mu.PreTrainedModel._load_pretrained_model.__func__ \
            if hasattr(_mu.PreTrainedModel._load_pretrained_model, '__func__') \
            else _mu.PreTrainedModel._load_pretrained_model

        from sentence_transformers import SentenceTransformer
        print("📦 Loading embedding model (first run only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("✓ Embedding model ready")
    return _model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, show_progress_bar=False).tolist()


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path="./qdrant_db")
        existing = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION not in existing:
            _qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    return _qdrant


def _csv_hash(csv_path: str) -> str:
    """MD5 of the CSV file — used to detect when data changes."""
    with open(csv_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_csv_to_qdrant(csv_path: str) -> None:
    """
    Ingest events CSV into Qdrant.
    Re-ingests automatically when the CSV file changes (hash check).
    """
    client = _get_qdrant()
    current_hash = _csv_hash(csv_path)

    # Store hash as a single dummy point payload for comparison
    hash_collection = "csv_meta"
    existing_cols = [c.name for c in client.get_collections().collections]

    if hash_collection not in existing_cols:
        client.create_collection(
            hash_collection,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )

    stored_hash = None
    try:
        results = client.scroll(hash_collection, limit=1)[0]
        if results:
            stored_hash = results[0].payload.get("hash")
    except Exception:
        pass

    if stored_hash == current_hash:
        count = client.count(COLLECTION).count
        print(f"✓ Qdrant up to date — {count} events loaded (CSV unchanged)")
        return

    # CSV changed or first run — reload
    print("📥 Loading events into Qdrant...")
    client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    df = pd.read_csv(csv_path)
    points = []
    for idx, row in df.iterrows():
        doc = (
            f"{row.get('event_name','')} {row.get('category','')} "
            f"{row.get('city','')} {row.get('country','')} "
            f"sponsors:{row.get('sponsors','')} "
            f"attendance:{row.get('attendance','')}"
        )
        points.append(
            PointStruct(id=int(idx), vector=_embed(doc), payload=row.to_dict())
        )

    client.upsert(collection_name=COLLECTION, points=points)

    # save new hash
    client.upsert(
        hash_collection,
        points=[PointStruct(id=0, vector=[0.0], payload={"hash": current_hash})],
    )
    print(f"✓ Loaded {len(points)} events into Qdrant")


def query_similar_events(category: str, geography: str, n: int = 3) -> list[dict]:
    """Return n past events most similar to category + geography."""
    try:
        client = _get_qdrant()
        if client.count(COLLECTION).count == 0:
            return []

        query_vec = _embed(f"{category} conference {geography}")

        # qdrant-client ≥1.10 renamed .search() → .query_points()
        try:
            hits = client.query_points(
                collection_name=COLLECTION,
                query=query_vec,
                limit=n,
            ).points
        except AttributeError:
            hits = client.search(
                collection_name=COLLECTION,
                query_vector=query_vec,
                limit=n,
            )

        return [hit.payload for hit in hits]
    except Exception as e:
        print(f"⚠️  Qdrant query failed: {e}")
        return []


# ─────────────────────────────────────────────
# JSON parser — 3-strategy fallback
# ─────────────────────────────────────────────
def parse_json(text: str, fallback):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\[[\s\S]*?\])", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    print(f"⚠️  All JSON parse strategies failed. Snippet: {text[:150]}")
    return fallback
