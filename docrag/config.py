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
