import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from hybrid import BM25Index, hybrid_search
from index import EMBED_MODEL, QDRANT_DIR
from search import search

EVAL_FILE = Path("eval/questions.json")
RESULTS_DIR = Path("eval/results")
MAX_K = 10  # retrieve more than we score, to see how far down a miss is


def first_relevant_rank(points, sources: list[str], evidence: list[str]) -> int | None:
    # 1-based rank of the first chunk from an accepted document containing any evidence string
    evidence = [e.lower() for e in evidence]
    for rank, point in enumerate(points, start=1):
        if point.payload["source"] not in sources:
            continue
        text = point.payload["text"].lower()
        if any(e in text for e in evidence):
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    summary = {"n": n}
    for k in (1, 3, 5):
        summary[f"hit@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    summary["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return summary


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    mode = sys.argv[2] if len(sys.argv) > 2 else "dense"  # dense | bm25 | hybrid

    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    questions = [q for q in questions if q["evidence"]]  # unanswerable ones have no evidence

    model = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(path=QDRANT_DIR)
    bm25 = BM25Index(qdrant)

    rows = []
    for q in questions:
        if mode == "dense":
            points = search(q["question"], model, qdrant, k=MAX_K)
        elif mode == "bm25":
            points = [point for point, _score in bm25.search(q["question"], k=MAX_K)]
        elif mode == "hybrid":
            points = hybrid_search(q["question"], model, qdrant, bm25, k=MAX_K)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        rank = first_relevant_rank(points, q["sources"], q["evidence"])
        rows.append({"id": q["id"], "lang": q["lang"], "category": q["category"],
                     "doc": q["sources"][0], "rank": rank})
        print(f"{q['id']}  {q['lang']}  {q['category']:<7} rank={str(rank or '-'):<3} {q['question'][:60]}")

    qdrant.close()

    # Overall score, plus per language, category and document
    groups = {"all": rows}
    for key in ("lang", "category", "doc"):
        for row in rows:
            groups.setdefault(f"{key}={row[key]}", []).append(row)
    summary = {name: summarize([r["rank"] for r in group]) for name, group in groups.items()}

    print(f"\n{'group':<20}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'mrr':>8}")
    for name, s in summary.items():
        print(f"{name:<20}{s['n']:>4}{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['hit@5']:>8.2f}{s['mrr']:>8.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"retrieval_{run_name}.json"
    out_file.write_text(json.dumps({"run": run_name, "mode": mode, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")