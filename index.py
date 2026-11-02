import uuid
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from chunks import build_chunks
from figures import add_descriptions
from parse import CACHE_DIR, parse_pdf

EMBED_MODEL = "BAAI/bge-m3"
QDRANT_DIR = "qdrant_data"
COLLECTION = "docs"


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
    client = QdrantClient(path=QDRANT_DIR)  # local mode: just a folder, no server

    # Rebuild the collection from scratch on every run
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    for pdf_path in sorted(Path("data").glob("*.pdf")):
        index_pdf(pdf_path, model, client)

    print(f"Indexed {client.count(COLLECTION).count} points into '{COLLECTION}'")
    client.close()