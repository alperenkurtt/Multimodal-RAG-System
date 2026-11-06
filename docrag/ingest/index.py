import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from docrag.config import CACHE_DIR, COLLECTION, DATA_DIR, EMBED_MODEL, QDRANT_DIR
from docrag.ingest.chunks import build_chunks
from docrag.ingest.figures import add_descriptions
from docrag.ingest.parse import parse_pdf



def chunk_point_id(source: str, chunk_id: str) -> str:
    # Qdrant ids must be int or UUID; uuid5 always gives the same id for the same input
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_id}"))


def index_pdf(pdf_path: Path, model: SentenceTransformer, client: QdrantClient) -> None:
    doc = parse_pdf(pdf_path)
    chunks = build_chunks(doc)
    add_descriptions(doc, chunks, CACHE_DIR / f"{pdf_path.stem}_figures.json")

    print(f"Embedding {len(chunks)} chunks from {pdf_path.name} ...")
    vectors = model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,  # unit length -> cosine similarity = dot product
        batch_size=8,
        show_progress_bar=True,
    )

    # The first PDF creates the collection; vector size comes from the model
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=chunk_point_id(pdf_path.name, chunk.id),
            vector=vector.tolist(),
            payload={**asdict(chunk), "source": pdf_path.name},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(COLLECTION, points=points)


if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)

    # Rebuild from scratch on every run. Remove the folder before opening the client:
    # on Windows delete_collection silently keeps the open SQLite file and its old points
    if Path(QDRANT_DIR).exists():
        shutil.rmtree(QDRANT_DIR)
    client = QdrantClient(path=QDRANT_DIR)  # local mode: just a folder, no server

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        index_pdf(pdf_path, model, client)

    print(f"Indexed {client.count(COLLECTION).count} points into '{COLLECTION}'")
    client.close()