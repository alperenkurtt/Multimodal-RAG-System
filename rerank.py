from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from index import EMBED_MODEL, QDRANT_DIR
from search import search

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATES = 20  # dense top-20 already contains every answer (recall@20 = 27/27)


def load_reranker() -> CrossEncoder:
    # Query + chunk are read together; longer pairs are truncated to 512 tokens
    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank_search(query, model, client, reranker: CrossEncoder, k: int = 10):
    candidates = search(query, model, client, k=CANDIDATES)
    pairs = [(query, point.payload["text"]) for point in candidates]
    scores = reranker.predict(pairs, batch_size=8)
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [(candidates[i], float(scores[i])) for i in order[:k]]


if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)
    reranker = load_reranker()
    client = QdrantClient(path=QDRANT_DIR)

    while True:
        query = input("\nQuestion (empty to quit): ").strip()
        if not query:
            break
        dense_ids = [p.id for p in search(query, model, client, k=CANDIDATES)]
        for point, score in rerank_search(query, model, client, reranker, k=5):
            p = point.payload
            dense_rank = dense_ids.index(point.id) + 1  # where dense alone had put it
            preview = p["text"][:70].replace("\n", " ")
            print(f"{score:.3f}  dense#{dense_rank:<3} {p['source']:<16} {p['id']:<11} {preview}")

    client.close()