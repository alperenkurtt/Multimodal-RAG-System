import re

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from docrag.config import COLLECTION, QDRANT_DIR
from docrag.retrieval.search import search


def tokenize(text: str) -> list[str]:
    # Lowercase word tokens; keeps numbers and ids like "80863" or "v5x6"
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, client: QdrantClient):
        # Read every chunk back from Qdrant, so dense and BM25 search the same corpus
        self.points, _ = client.scroll(COLLECTION, limit=10_000, with_payload=True)
        self.bm25 = BM25Okapi([tokenize(p.payload["text"]) for p in self.points])

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.points[i], scores[i]) for i in best]


def rrf(rankings: list[list], k: int = 60) -> list:
    # Reciprocal Rank Fusion: only ranks matter, raw scores are ignored
    scores: dict = {}
    points: dict = {}
    for ranking in rankings:
        for rank, point in enumerate(ranking, start=1):
            scores[point.id] = scores.get(point.id, 0.0) + 1 / (k + rank)
            points[point.id] = point
    order = sorted(scores, key=scores.get, reverse=True)
    return [points[point_id] for point_id in order]


def hybrid_search(query, model, client, bm25: BM25Index, k: int = 10, candidates: int = 20):
    # Each method proposes its top candidates, RRF merges them
    dense = search(query, model, client, k=candidates)
    sparse = [point for point, _score in bm25.search(query, k=candidates)]
    return rrf([dense, sparse])[:k]


if __name__ == "__main__":
    client = QdrantClient(path=QDRANT_DIR)
    index = BM25Index(client)
    print(f"BM25 index over {len(index.points)} chunks")

    while True:
        query = input("\nQuestion (empty to quit): ").strip()
        if not query:
            break
        for point, score in index.search(query, k=5):
            p = point.payload
            preview = p["text"][:80].replace("\n", " ")
            print(f"{score:6.2f}  {p['source']:<16} {p['id']:<11} {preview}")

    client.close()