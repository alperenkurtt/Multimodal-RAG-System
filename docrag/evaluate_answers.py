import json
import re
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from docrag.answer import TOP_K, answer, retrieve
from docrag.config import DATA_DIR, EMBED_MODEL, JUDGE_MODEL, QDRANT_DIR
from docrag.evaluate import EVAL_FILE, RESULTS_DIR, first_relevant_rank
from docrag.ingest.figures import client as llm
from docrag.ingest.parse import parse_pdf
from docrag.retrieval.rerank import load_reranker

JUDGE_PROMPT = """You grade an answer produced by a document QA system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Grade only the facts, not style or language (the system may answer in Turkish or English).
- "correct": contains the key facts of the reference answer, nothing contradicts it.
- "partial": some key facts right, but something important is missing or wrong.
- "wrong": key facts wrong or missing.
If the reference says the information is not in the document, the answer is "correct" only if
the system says it cannot find it, and "wrong" if it answers anyway.

Reply with JSON only: {{"verdict": "correct" | "partial" | "wrong", "reason": "<one sentence>"}}"""


def judge(question: str, reference: str, system_answer: str) -> dict:
    response = llm.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=2000,  # reasoning tokens count toward this budget too
        temperature=0,
        extra_body={"reasoning": {"effort": "low"}},  # mandatory for this model, keep it short
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=system_answer)}],
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(f"Incomplete judge response: finish_reason={choice.finish_reason!r}")
    # Models sometimes wrap JSON in ```json fences; keep only the {...} part
    return json.loads(re.search(r"\{.*\}", choice.message.content, re.DOTALL).group())


def cited_points(answer_text: str, points) -> list:
    # "[2][4]" -> the 2nd and 4th retrieved chunks; ids out of range are ignored
    ids = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    return [points[i - 1] for i in sorted(ids) if 1 <= i <= len(points)]


def rate(rows: list[dict], condition) -> float:
    return round(sum(condition(r) for r in rows) / len(rows), 2)


if __name__ == "__main__":
    run_name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "rerank"  # dense | rerank

    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    docs = {pdf.name: parse_pdf(pdf) for pdf in DATA_DIR.glob("*.pdf")}
    model = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(path=QDRANT_DIR)
    reranker = load_reranker() if mode == "rerank" else None

    rows = []
    for q in questions:
        points = [point for point, _ in retrieve(q["question"], model, qdrant, reranker, k=TOP_K)]
        text = answer(q["question"], points, docs)
        grade = judge(q["question"], q["answer"], text)

        row = {"id": q["id"], "lang": q["lang"], "category": q["category"],
                "verdict": grade["verdict"], "reason": grade["reason"], "answer": text}
        if q["evidence"]:
            # Was the answer in the context at all, and does a cited chunk contain it?
            row["in_context"] = first_relevant_rank(points, q["sources"], q["evidence"]) is not None
            row["cited_ok"] = first_relevant_rank(cited_points(text, points), q["sources"], q["evidence"]) is not None
        rows.append(row)
        print(f"{q['id']}  {grade['verdict']:<8} ctx={str(row.get('in_context', '-')):<6}"
            f"cite={str(row.get('cited_ok', '-')):<6} {grade['reason'][:70]}")

    qdrant.close()

    answerable = [r for r in rows if "in_context" in r]
    unanswerable = [r for r in rows if "in_context" not in r]
    summary = {
        "n_answerable": len(answerable),
        "correct": rate(answerable, lambda r: r["verdict"] == "correct"),
        "correct_or_partial": rate(answerable, lambda r: r["verdict"] != "wrong"),
        "in_context": rate(answerable, lambda r: r["in_context"]),
        "cited_ok": rate(answerable, lambda r: r["cited_ok"]),
        "n_unanswerable": len(unanswerable),
        "refused": rate(unanswerable, lambda r: r["verdict"] == "correct"),
    }
    print("\n" + json.dumps(summary, indent=2))

    out_file = RESULTS_DIR / f"answers_{run_name}.json"
    out_file.write_text(json.dumps({"run": run_name, "mode": mode, "summary": summary, "rows": rows},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_file}")