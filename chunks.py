import json
from dataclasses import asdict, dataclass
from pathlib import Path

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

from parse import CACHE_DIR, parse_pdf

MAX_CHARS = 1500        # max size of a text chunk (~350 tokens)
MIN_PICTURE_SIDE = 200  # skip logos / tiny images (pixels, at images_scale=2)


@dataclass
class Chunk:
    id: str
    type: str                  # "text" | "table" | "picture"
    text: str                  # content that will be embedded
    pages: list[int]
    heading: str | None = None
    ref: str | None = None     # docling self_ref, e.g. "#/pictures/1"


def build_chunks(doc: DoclingDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = None
    buffer: list[str] = []
    buffer_pages: set[int] = set()

    def flush():
        # citation lists add noise to retrieval
        if heading == "References":
            buffer.clear()
            buffer_pages.clear()
            return
        # Turn the buffered paragraphs into one text chunk
        if not buffer:
            return
        body = "\n".join(buffer)
        text = f"{heading}\n\n{body}" if heading else body
        chunks.append(Chunk(
            id=f"text-{len(chunks)}",
            type="text",
            text=text,
            pages=sorted(buffer_pages),
            heading=heading,
        ))
        buffer.clear()
        buffer_pages.clear()

    # Reading order; skips text inside pictures and page headers/footers
    for item, _level in doc.iterate_items():
        page = item.prov[0].page_no if item.prov else None

        if isinstance(item, SectionHeaderItem):
            flush()
            heading = item.text

        elif isinstance(item, TableItem):
            if item.data.num_rows == 0:  # empty / misdetected table
                continue
            chunks.append(Chunk(
                id=f"table-{len(chunks)}",
                type="table",
                text=item.export_to_markdown(doc),  # includes the caption
                pages=[page],
                heading=heading,
                ref=item.self_ref,
            ))

        elif isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is None or min(image.size) < MIN_PICTURE_SIDE:
                continue
            chunks.append(Chunk(
                id=f"picture-{len(chunks)}",
                type="picture",
                text=item.caption_text(doc),  # VLM description is added in the next step
                pages=[page],
                heading=heading,
                ref=item.self_ref,
            ))

        elif isinstance(item, TextItem):
            # Captions of tables/pictures are already in their own chunk
            is_attached_caption = (
                item.label == DocItemLabel.CAPTION
                and item.parent.cref.startswith(("#/tables", "#/pictures"))
            )
            if is_attached_caption:
                continue

            buffer_size = sum(len(t) for t in buffer)
            if buffer_size + len(item.text) > MAX_CHARS:
                flush()
            buffer.append(item.text)
            buffer_pages.add(page)

    flush()
    return chunks


if __name__ == "__main__":
    pdf_path = Path("data/arxiv.pdf")
    doc = parse_pdf(pdf_path)
    chunks = build_chunks(doc)

    # Save for inspection
    out_file = CACHE_DIR / f"{pdf_path.stem}_chunks.json"
    out_file.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {len(chunks)} chunks to {out_file}\n")

    for c in chunks:
        preview = c.text[:70].replace("\n", " ")
        print(f"{c.id:<12} pages={c.pages} heading={c.heading!r}\n    {preview!r}")