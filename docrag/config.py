from pathlib import Path

# Paths are relative to the project root: always run from there with `python -m docrag...`
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
QDRANT_DIR = "qdrant_data"
COLLECTION = "docs"

# Models
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "anthropic/claude-haiku-5.5"  # answers
VLM_MODEL = "anthropic/claude-haiku-5.5"  # figure descriptions
JUDGE_MODEL = "anthropic/claude-sonnet-5.5" # grades answers

# Pages that are screenshots of another document in the corpus
SKIP_PAGES = {"arxiv": {8, 9}}

# Captions lost by the parser, copied from the PDF text layer
MISSING_CAPTIONS = {
    "doclaynet": {
        "#/tables/4": (
            "Table 5: Prediction Performance (mAP@0.5-0.95) of a Mask R-CNN R50 network "
            "across the PubLayNet, DocBank & DocLayNet data-sets. By evaluating on common "
            "label classes of each dataset, we observe that the DocLayNet-trained model "
            "has much less pronounced variations in performance across all datasets."
        ),
    },
}
