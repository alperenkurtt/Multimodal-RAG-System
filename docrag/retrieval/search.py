from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from docrag.config import COLLECTION, EMBED_MODEL, QDRANT_DIR


def search(query: str, model: SentenceTransformer, client: QdrantClient, k: int = 5):
    query_vector = model.encode(query, normalize_embeddings=True)
    return client.query_points(COLLECTION, query=query_vector.tolist(), limit=k).points


if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_DIR)

    # Loop so the model is loaded only once
    while True:
        query = input("\nQuestion (empty to quit): ").strip()
        if not query:
            break
        for point in search(query, model, client):
            p = point.payload
            preview = p["text"][:100].replace("\n", " ")
            print(f"{point.score:.3f}  {p['type']:<8} pages={p['pages']}  {p['heading']!r}")
            print(f"       {preview}")

    client.close()