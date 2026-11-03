
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc import RefItem
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from docrag.config import DATA_DIR, EMBED_MODEL, LLM_MODEL, QDRANT_DIR
from docrag.ingest.figures import client as llm
from docrag.ingest.figures import image_to_data_url
from docrag.ingest.parse import parse_pdf
from docrag.retrieval.search import search

TOP_K = 5

SYSTEM_PROMPT = """You answer questions about documents using only the numbered sources provided.

Rules:
- Use only information from the sources. If they don't contain the answer, say so clearly.
- Cite every claim with the id of the source it comes from, in brackets, e.g. [2]. Use several ids if needed, e.g. [1][3]. Only cite a source if itactually contains that claim.
- Some sources are figures: you get a text description and the original image. Use both.
- Answer in the same language as the question.
- Be concise."""


def build_context(points, docs: dict[str, DoclingDocument]) -> list[dict]:
    # One text part per source; picture sources also get their image
    parts = []
    for n, point in enumerate(points, start=1):
        p = point.payload
        header = f"{p['source']} | {p['type']} | pages {p['pages']} | section: {p['heading']}"
        parts.append({"type": "text", "text": f'<source id="{n}">\n{header}\n{p["text"]}'})

        if p["type"] == "picture":
            doc = docs[p["source"]]
            image = RefItem(cref=p["ref"]).resolve(doc).get_image(doc)
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})

        # Close the tag after the image, so the image clearly belongs to this source
        parts.append({"type": "text", "text": "</source>"})
    return parts


def answer(question: str, points, docs: dict[str, DoclingDocument]) -> str:
    content = build_context(points, docs)
    content.append({"type": "text", "text": f"Question: {question}"})

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=2000,
        temperature=0,
        extra_body={"reasoning": {"enabled": False}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(f"Incomplete LLM response: finish_reason={choice.finish_reason!r}")
    return choice.message.content.strip()


if __name__ == "__main__":
    # Parsed docs are needed to load figure images (loaded from cache, fast)
    docs = {pdf.name: parse_pdf(pdf) for pdf in DATA_DIR.glob("*.pdf")}
    model = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(path=QDRANT_DIR)

    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break

        points = search(question, model, qdrant, k=TOP_K)
        print("\n" + answer(question, points, docs))

        print("\nSources:")
        for n, point in enumerate(points, start=1):
            p = point.payload
            print(f"  [{n}] {p['source']} | {p['type']} | pages {p['pages']} | {p['heading']!r} (score {point.score:.3f})")

    qdrant.close()