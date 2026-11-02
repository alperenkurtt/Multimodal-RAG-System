import base64
import io
import json
import os
import re
from pathlib import Path

from docling_core.types.doc import DoclingDocument, PictureItem, RefItem, TextItem
from dotenv import load_dotenv
from openai import OpenAI
from PIL.Image import Image

from chunks import Chunk, build_chunks
from parse import CACHE_DIR, parse_pdf

load_dotenv()

VLM_MODEL = "anthropic/claude-haiku-5.5"

PROMPT = """You are describing a figure from a document so that it can be found by text search.

Context from the document:
- Section: {heading}
- Caption: {caption}

Text extracted from the PDF inside this figure (characters are exact, but the order may be scrambled):
<<<
{text_layer}
>>>

Describe the figure:
1. What kind of figure it is (chart, diagram, table screenshot, code, page screenshot, ...).
2. Its main message or purpose.
3. All important visible text: labels, axis titles, legend entries, numbers, names.
4. If it contains a table or code, reproduce it in full: every row and every column.

When you write numbers or labels, copy them exactly from the extracted text above;
use the image to understand layout and structure. If the extracted text is "none",
read from the image. Be factual. Do not guess values you cannot read."""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def image_to_data_url(image: Image) -> str:
    # PIL image -> PNG bytes -> base64 string
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def picture_text_layer(doc: DoclingDocument, picture: PictureItem) -> str:
    # Text Docling found inside the picture (PDF text layer, not OCR)
    texts = [
        item.text
        for item, _ in doc.iterate_items(root=picture, traverse_pictures=True)
        if isinstance(item, TextItem)
    ]
    return "\n".join(texts)


def unverified_numbers(description: str, text_layer: str) -> list[str]:
    # Numbers in the description that don't appear in the text layer
    # (2+ characters, to skip list markers like "1." "2.")
    pattern = r"\d+(?:\.\d+)?(?:-\d+)?"
    known = set(re.findall(pattern, text_layer))
    return [n for n in re.findall(pattern, description) if len(n) >= 2 and n not in known]


def describe_image(image: Image, heading: str | None, caption: str, text_layer: str) -> str:
    prompt = PROMPT.format(
        heading=heading or "unknown",
        caption=caption or "none",
        text_layer=text_layer or "none",
    )
    response = client.chat.completions.create(
        model=VLM_MODEL,
        max_tokens=3000,
        temperature=0,
        extra_body={"reasoning": {"enabled": False}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ],
        }],
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        # Don't return (and cache) a truncated or failed answer
        raise RuntimeError(
            f"Incomplete VLM response: finish_reason={choice.finish_reason!r}, "
            f"usage={response.usage}"
        )
    return choice.message.content.strip()


def add_descriptions(doc: DoclingDocument, chunks: list[Chunk], cache_file: Path) -> None:
    # Fills picture chunks with "caption + VLM description", using a disk cache
    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    for chunk in chunks:
        if chunk.type != "picture":
            continue

        caption = chunk.text
        if chunk.ref not in cache:
            print(f"Describing {chunk.ref} (page {chunk.pages[0]}) ...")
            picture = RefItem(cref=chunk.ref).resolve(doc)
            text_layer = picture_text_layer(doc, picture)
            description = describe_image(picture.get_image(doc), chunk.heading, caption, text_layer)

            if text_layer and text_layer.strip() != caption.strip():
                suspicious = unverified_numbers(description, text_layer)
                if suspicious:
                    print(f"  WARNING: numbers not found in text layer: {suspicious}")

            cache[chunk.ref] = description
            # Save after every call so nothing is lost if a later call fails
            cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

        description = cache[chunk.ref]
        chunk.text = f"{caption}\n\n{description}" if caption else description


if __name__ == "__main__":
    pdf_path = Path("data/arxiv.pdf")
    doc = parse_pdf(pdf_path)
    chunks = build_chunks(doc)
    add_descriptions(doc, chunks, CACHE_DIR / f"{pdf_path.stem}_figures.json")

    for c in chunks:
        if c.type == "picture":
            print(f"\n===== {c.id} | page {c.pages} | {c.ref} =====")
            print(c.text)