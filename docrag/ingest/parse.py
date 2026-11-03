from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from docrag.config import CACHE_DIR, DATA_DIR


def parse_pdf(pdf_path: Path) -> DoclingDocument:
    cache_file = CACHE_DIR / f"{pdf_path.stem}.json"

    # Load from disk if already parsed
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        return DoclingDocument.load_from_json(cache_file)

    options = PdfPipelineOptions()
    options.do_ocr = False                  # digital PDF, OCR not needed
    options.do_table_structure = True       # extract table cell structure
    options.generate_picture_images = True  # keep images of figures
    options.images_scale = 2.0              # sharper images for the VLM

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    print(f"Parsing: {pdf_path}")
    doc = converter.convert(pdf_path).document

    CACHE_DIR.mkdir(exist_ok=True)
    doc.save_as_json(cache_file)
    print(f"Saved: {cache_file}")
    return doc


if __name__ == "__main__":
    doc = parse_pdf(DATA_DIR / "arxiv.pdf")

    print(f"\nText items : {len(doc.texts)}")
    print(f"Tables     : {len(doc.tables)}")
    print(f"Pictures   : {len(doc.pictures)}")

    print("\n--- Tables ---")
    for i, table in enumerate(doc.tables):
        page = table.prov[0].page_no
        print(f"\n[Table {i}] page {page} | caption: {table.caption_text(doc)!r}")
        print(table.export_to_markdown(doc)[:300])

    print("\n--- Pictures ---")
    for i, pic in enumerate(doc.pictures):
        page = pic.prov[0].page_no
        img = pic.get_image(doc)
        size = img.size if img else None
        print(f"[Picture {i}] page {page} | size: {size} | caption: {pic.caption_text(doc)!r}")