
"""Central configuration for the Medical RAG project."""

from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

# config.py is located in PROJECT_ROOT/03_CODE/src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset folders
DATASETS = PROJECT_ROOT / "05_DATASETS"
PDF_DIR = DATASETS / "pdfs"
PROCESSED_DIR = DATASETS / "processed"

# Extra project outputs
RESULTS_DIR = PROJECT_ROOT / "07_EXTRA" / "results"


# --------------------------------------------------
# Embedding model / tokenizer
# --------------------------------------------------

EMBED_MODEL = "BAAI/bge-m3"


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

CHUNK_TOKENS = 512
CHUNK_OVERLAP = 102

# Keep every non-empty page initially.
# We can adjust this after inspecting the ingestion results.
MIN_PAGE_TOKENS = 1

# Maximum allowed number of chunks for this project
MAX_CHUNKS = 4500
