# Multimodal PDF RAG

Question answering over PDFs that contain text, tables and figures. Answers cite
their sources (document, page, table, figure), and retrieval quality is measured
with a hand-written evaluation set.

Built from scratch for learning: no LangChain / LlamaIndex, no servers, terminal only.
Questions can be asked in any language (tested with English and Turkish).

## Pipeline

```
PDF ─► parse.py ─► chunks.py ─► figures.py ─► index.py ─► search.py / hybrid.py / rerank.py ─► answer.py
       Docling     text/table/   VLM figure    bge-m3 +    dense / BM25 / RRF / reranker      LLM answer
                   picture       descriptions  Qdrant                                         with citations
                                                                 │
                                                            evaluate.py
```

```
data/            input PDFs
cache/           parsed documents + figure descriptions (generated)
qdrant_data/     vector index (generated)
eval/            questions.json + results/
docrag/
├── config.py    paths and model names
├── ingest/      parse.py, chunks.py, figures.py, index.py   (PDF -> vectors, run once)
├── retrieval/   search.py, hybrid.py, rerank.py             (question -> chunks)
├── answer.py    chunks -> cited answer
└── evaluate.py  retrieval metrics
```

| Step | File | What it does |
|---|---|---|
| Parse | `ingest/parse.py` | Docling: layout, text, table structure (TableFormer), figure images. Cached as JSON in `cache/`. |
| Chunk | `ingest/chunks.py` | Three chunk types: **text** (grouped under section headings, max ~1500 chars), **table** (markdown + caption), **picture**. Skips references and tiny images. |
| Figures | `ingest/figures.py` | Sends each figure to a VLM (Claude Haiku 5.5 via OpenRouter) for a searchable description. The PDF text layer is added to the prompt so numbers are copied, not guessed. Numbers not found in the text layer are flagged. Descriptions are cached. |
| Index | `ingest/index.py` | Embeds all chunks of every PDF in `data/` with **bge-m3** and stores them in a local **Qdrant** folder, with type / page / heading / source metadata. |
| Search | `retrieval/search.py`, `retrieval/hybrid.py` | Dense cosine search; BM25 (`rank_bm25`); Reciprocal Rank Fusion of both. |
| Rerank | `retrieval/rerank.py` | Dense top-20 candidates re-sorted by a cross-encoder (**bge-reranker-v2-m3**) that reads query + chunk together. |
| Answer | `answer.py` | Top chunks become numbered sources; figure sources also send the original image. The LLM answers in the question's language and cites sources like `[1][3]`. |
| Eval | `evaluate.py` | hit@1/3/5 and MRR, grouped by language, category and document. |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENROUTER_API_KEY=...
```

## Usage

```powershell
# always run from the project root
python -m docrag.ingest.index                    # parse + describe figures + embed all PDFs in data/
python -m docrag.retrieval.search                # interactive dense search
python -m docrag.retrieval.hybrid                # interactive BM25 search
python -m docrag.retrieval.rerank                # interactive dense top-20 + rerank
python -m docrag.answer                          # interactive cited answers
python -m docrag.evaluate <run_name> <mode>      # mode: dense | bm25 | hybrid | rerank
```

## Corpus

| File | Document | Chunks |
|---|---|---|
| `data/arxiv.pdf` | Docling Technical Report | 26 |
| `data/doclaynet.pdf` | DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis | 46 |
| `data/tableformer.pdf` | TableFormer: Table Structure Understanding with Transformers | 59 |

The two papers are cited by the Docling report, so they were chosen on purpose
as **hard negatives**: similar topic and vocabulary, different facts.

## Evaluation

`eval/questions.json` holds 30 questions:

- **Languages:** English and Turkish (the documents are English).
- **Categories:** text, table, figure (answer only visible in an image), trap, unanswerable.
- **Labels:** each question has `evidence` (strings that must appear in a relevant chunk),
  `sources` (documents allowed to answer it), and a reference `answer`.

A retrieved chunk counts as a hit only if it comes from an accepted source **and**
contains one of the evidence strings. Unanswerable questions are skipped in
retrieval eval; they are kept for the later answer eval. Results are saved in `eval/results/`.

### Results (retrieval)

| Run | Corpus | Questions | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|
| `baseline`: dense | 1 PDF | 14 | 0.86 | 1.00 | 1.00 | 0.92 |
| `multi_doc`: dense (baseline) | 3 PDFs | 27 | 0.74 | **0.89** | 0.89 | 0.81 |
| `bm25`: BM25 only | 3 PDFs | 27 | 0.59 | 0.74 | 0.78 | 0.67 |
| `hybrid`: dense + BM25, RRF (k=60) | 3 PDFs | 27 | 0.59 | 0.78 | 0.78 | 0.70 |
| **`rerank`: dense top-20 → bge-reranker-v2-m3 (current best)** | 3 PDFs | 27 | **0.78** | 0.85 | **0.96** | **0.85** |

Dense baseline by group:

| Group | n | hit@1 | MRR |
|---|---|---|---|
| English | 17 | 0.76 | 0.83 |
| Turkish | 10 | 0.70 | 0.78 |
| text | 14 | 0.79 | 0.86 |
| table | 7 | 0.57 | 0.64 |
| figure | 5 | 0.80 | 0.90 |
| arxiv.pdf | 14 | 0.93 | 0.95 |
| doclaynet.pdf | 6 | 0.33 | 0.47 |
| tableformer.pdf | 7 | 0.71 | 0.83 |

### What changed the numbers, and why

**1 → 3 documents (hit@1 0.86 → 0.74).** The new questions are harder, and the new
papers compete with each other. The two runs use different question sets, so this
change is not a like-for-like comparison. Weak spots: tables (0.57) and DocLayNet (0.33).
Most misses come from the right document but the wrong chunk. "DocLayNet" appears in
almost every chunk of that paper, so a single dense vector cannot tell them apart.

**Eval labels went stale (fixed).** After adding DocLayNet, q13 ("authors of the
DocLayNet paper") was answered by the real paper, but the label only accepted the
Docling report, so it counted as a miss. Fix: `source` became a `sources` list.
Two questions (q18, q27) got an extra evidence string after checking that the
retrieved chunk really answers them. q19 was kept strict on purpose: adding
evidence just because the system found something would make the eval meaningless.
These fixes took dense from 0.70 to 0.74 hit@1.

**BM25 alone (MRR 0.67, lower).**
- Better on exact rare terms: q17 "production annotation" rank 7 → 3, q27 rank 3 → 1.
- Worse on Turkish questions (MRR 0.54), which have no shared words with English text.
- Still good on Turkish questions that contain names (OCR, Intel Xeon, Camelot, TEDS).
- Misses tables whose words differ from the question ("best overall" vs "All").

**Hybrid RRF (MRR 0.70, lower than dense).**

| Question | Dense rank | Hybrid rank |
|---|---|---|
| q17 | 7 | **1** |
| q27 | 3 | **1** |
| q01 | 1 | 3 |
| q02 | 3 | 9 |
| q08 | 1 | 2 |
| q12 | 1 | 7 |
| q13 | 1 | 6 |
| q18 | 1 | 6 |
| q19 | 6 | 8 |
| q24 | 2 | 3 |
| q25 | 1 | 2 |

Why it got worse:
- RRF rewards chunks that appear in **both** lists. With k=60, rank 1 and rank 5 score
  almost the same, so generic chunks found at middle ranks by both methods beat
  the dense rank-1 answer when BM25 ranks it low. Example: q12's answer was
  dense #1 but BM25 #14, and ended at #7.
- On Turkish questions the BM25 list is mostly noise, but it gets equal weight.
- A small sweep over RRF k, BM25 weight and BM25 candidate count never beat dense.
- **Key finding:** dense top-20 already contains the answer for 27/27 questions,
  and adding BM25 candidates does not raise that. The problem is **ordering, not
  recall**, so the next step is a reranker, not more retrievers. Dense stays the default.

**Reranker (MRR 0.81 → 0.85, hit@5 0.89 → 0.96).** Dense retrieves 20 candidates,
then `bge-reranker-v2-m3` (cross-encoder) reads each query + chunk pair together and re-sorts them.

| Question | Dense rank | Rerank rank |
|---|---|---|
| q17 | 7 | **4** |
| q19 | 6 | **5** |
| q20 | – | **6** |
| q24 | 2 | **1** |
| q27 | 3 | **1** |
| q02 | 3 | 4 |
| q25 | 1 | 2 |

- Main gain is pulling deep answers into the top 5 (the chunks the LLM sees).
- hit@3 dropped by one question (q02). With 27 questions, one question ≈ 0.04,
  so the gain is real but small.
- q17: the fact is at char 139 of its chunk, so 512-token truncation is not the cause.
  The reranker scores a topically close chunk without the answer higher (0.975 vs 0.775).
- Scores saturate near 1.0 for related chunks, and duplicates get identical scores:
  the arxiv screenshot copies still take slots. DocLayNet (hit@1 0.33) is now a
  data problem (duplicates, detached caption), not an ordering problem.
- Cost: every query runs 20 cross-encoder passes on CPU (much slower than dense).

**Figure check false alarms (fixed).** In the TableFormer appendix, tables are
embedded as images, so the text layer holds only the caption and every number
was flagged. After checking by eye, the VLM values were correct. The check now
skips figures whose text layer is just the caption.

### Known issues

- The Docling report (p. 8) contains a screenshot of the DocLayNet paper. Its text
  duplicates the real paper and takes top-k slots (q19). Needs deduplication.
- DocLayNet Table 5's caption was parsed into a nearby text chunk, so the table
  chunk is numbers only (q20: not in dense top 10, rank 6 after rerank).
- TableFormer Table 1: check marks were parsed as `3` / `7`.
- TableFormer: the sub-caption "a. Picture of a table:" was detected as a section
  heading, so some intro chunks have the wrong heading.
- Long chunks dilute single facts (q17's chunk is 2261 chars; the average is ~1200).

## Roadmap

- [x] Parse, chunk, VLM figure descriptions, index, cited multimodal answers
- [x] Retrieval eval, multi-document baseline
- [x] BM25 + RRF hybrid (measured: did not help on this data)
- [x] Reranker (`bge-reranker-v2-m3`) on dense top-20 (MRR 0.81 → 0.85)
- [ ] Duplicate removal, re-attach detached table captions, chunk size experiments
- [ ] Answer and citation eval (is the answer correct, and does the cited source contain it)
- [ ] Vision-based retrieval comparison (ColQwen-style)

## Stack

Docling · sentence-transformers (BAAI/bge-m3, BAAI/bge-reranker-v2-m3) · Qdrant (local mode) · rank_bm25 ·
OpenRouter (Claude Haiku 5.5 for VLM and answers) · Python 3
