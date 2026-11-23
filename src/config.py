
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


# --------------------------------------------------
# Vector store (Chroma)
# --------------------------------------------------

CHROMA_DIR = PROJECT_ROOT / "07_EXTRA" / "chroma_db"
COLLECTION = "medical"

# --------------------------------------------------
# Retrieval
# --------------------------------------------------

TOP_K = 5

#--------------------------------------------------
# LLM configuration
#--------------------------------------------------
DEFAULT_LLM = "llama3"
LLM_TEMPERATURE = 0
LLM_SEED = 42
LLM_NUM_CTX = 4096          # explicit: Ollama's default silently truncates the prompt
LLM_TIMEOUT_S = 120
OPENAI_MAX_RETRIES = 2

DISCLAIMER = "This is not medical advice. Consult a healthcare professional."
REFUSAL_SENTENCE = "I cannot answer this question from the provided documents."