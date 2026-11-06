# Multimodal PDF RAG

Ask questions about PDFs that contain text, tables and figures. Answers cite their
sources, and quality is measured with a hand-written evaluation set.

Built from scratch for learning: no LangChain or LlamaIndex. Runs in the terminal.
Questions can be in English or Turkish.

## How it works

1. **Parse:** Docling splits each PDF into text, tables and images.
2. **Chunk:** content becomes text, table and picture chunks.
3. **Describe:** a vision model writes a searchable description of each figure.
4. **Index:** bge-m3 embeds the chunks into a local Qdrant folder.
5. **Search:** dense search finds 20 candidates, bge-reranker-v2-m3 re-sorts them.
6. **Answer:** the top 5 chunks go to the LLM, which answers with citations like `[1][3]`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with `OPENROUTER_API_KEY=...`, and put your PDFs in `data/`.

## Usage

Run from the project root:

```powershell
python -m docrag.ingest.index                        # build the index
python -m docrag.answer                              # ask questions
python -m docrag.evaluate <run> <mode> [file]        # retrieval eval: dense | bm25 | hybrid | rerank
python -m docrag.evaluate_answers <run> <mode> [file] # answer eval: dense | rerank (paid LLM calls)
```

## Results

The corpus is 3 related papers (125 chunks): the Docling report, DocLayNet and TableFormer.

There are two question sets:

- `eval/questions.json`: 30 questions, 27 with a retrievable answer. Used while building the system.
- `eval/questions_holdout.json`: 20 new questions, 16 scored. Not used for tuning.

**Retrieval, main set**

| Method | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| BM25 (before cleanup) | 0.59 | 0.74 | 0.78 | 0.67 |
| Dense + BM25 hybrid (before cleanup) | 0.59 | 0.78 | 0.78 | 0.70 |
| Dense (before cleanup) | 0.74 | 0.89 | 0.89 | 0.81 |
| Dense + rerank (before cleanup) | 0.78 | 0.85 | 0.96 | 0.85 |
| Dense | 0.78 | 0.96 | 0.96 | 0.86 |
| **Dense + rerank** | **0.81** | **0.96** | **1.00** | **0.89** |

**Retrieval, held-out set**

| Method | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Dense | 0.56 | 0.81 | 0.94 | 0.70 |
| **Dense + rerank** | **0.69** | **0.88** | 0.88 | **0.79** |

**Answers, main set** (graded by an LLM judge)

| Method | Correct | Correct or partial | Citation ok | Refused unanswerable |
|---|---|---|---|---|
| Dense | 0.89 | 0.96 | 0.96 | 3/3 |
| **Dense + rerank** | **0.96** | **1.00** | **1.00** | 3/3 |

## What I learned

- **Hybrid search made things worse.** Dense top-20 already held every answer, so the
  problem was ordering, not finding. That is why I added a reranker.
- **Cleaning the data helped as much as the reranker on the main set.** Removing duplicate
  pages and restoring a lost table caption raised dense MRR from 0.81 to 0.86.
- **Held-out scores are lower.** The main set was used during development, so its numbers
  are optimistic. The cleanup did not change the held-out overall scores.
- **Eval labels can be wrong.** Reading failed answers found label errors, not only system errors.

## Limitations

- The question sets are small: one question is about 0.04 on the main set.
- The reranker helps English but not Turkish on the held-out set (MRR 0.85 vs dense 0.66
  in English, 0.71 vs 0.74 in Turkish).
- The duplicate pages and the lost caption are fixed by hand in `docrag/config.py`, so
  new PDFs do not get these fixes automatically.
- The answer eval was not run on the held-out set.
- Reranking is slow on CPU.

## Future work

- Compare with vision-based retrieval (ColQwen style).
- Translate Turkish questions to English before reranking.
- Try smaller chunks.

## Stack

Docling, sentence-transformers (bge-m3, bge-reranker-v2-m3), Qdrant local mode, rank_bm25,
OpenRouter (Claude Haiku 5.5 for figures and answers, Claude Sonnet 5.5 as judge).
