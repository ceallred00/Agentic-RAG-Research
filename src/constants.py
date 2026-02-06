from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

env_root = os.getenv("PROJECT_ROOT", "./")
# Wrap env_root in Path for easier path manipulations
ROOT_DIR = Path(env_root).resolve()

CONFIGS_DIR = ROOT_DIR / "configs"
SRC_DIR = ROOT_DIR / "src"

LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "app.log"

BASE_DIAGRAM_DIR = ROOT_DIR / "diagrams"
PROD_DIAGRAM_DIR = BASE_DIAGRAM_DIR / "production"

DATA_DIR = ROOT_DIR/ "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UWF_PUBLIC_KB_PROCESSED_DATE_DIR = PROCESSED_DATA_DIR / "uwf_public_kb"

# --- Embedding Model Constraints ---
# Token Approximation: 1 token ~= 4 chars. 2048 * 4 = 8192 chars.
    # Ref: http://ai.google.dev/gemini-api/docs/tokens?authuser=1&lang=python
# We set a safe buffer at 8000 to avoid API errors. 

# Model: gemini-embedding-001
# Limit: 2048 tokens. 
    # Ref: https://ai.google.dev/gemini-api/docs/embeddings?authuser=1#model-versions
GEMINI_EMBEDDING_MAX_CHAR_LIMIT = 8000
# Doc Ref: https://reference.langchain.com/python/integrations/langchain_google_genai/GoogleGenerativeAIEmbeddings/#langchain_google_genai.GoogleGenerativeAIEmbeddings.output_dimensionality
GEMINI_EMBEDDING_BATCH_LIMIT = 100

# Model: pinecone-sparse-english-v0
# Limit: 2048 tokens.
    # Ref: https://docs.pinecone.io/models/pinecone-sparse-english-v0
PINECONE_EMBEDDING_MAX_CHAR_LIMIT = 8000
PINECONE_MAX_BATCH_SIZE = 96
# Max batch size for upserts to Pinecone index to avoid request size limits (2 MB).
    # Ref: https://docs.pinecone.io/guides/index-data/upsert-data
# Vectors are heavy on size due to text in metadata and sparse representation. 
PINECONE_UPSERT_MAX_BATCH_SIZE = 50

# --- Chunking Strategies ---
# Chunk size set at 2000 chars (approximately 500 words) to ideally capture full procedural contexts in knowledge base.
# Overlap set to 20% (400 chars) to ensure that contextual information is carried between sections as needed.
CHUNKING_SIZE = 2000
CHUNKING_OVERLAP = 400

FAKE_DEPARTMENT_ADVISORS = {
    "Computer Science": {"name": "Dr. Smith", "email": "jane.smith@uwf.edu"}, 
    "Mathematics": {"name": "Dr. Johnson", "email": "john.johnson@uwf.edu"},
    "Biology": {"name": "Dr. Lee", "email": "sarah.lee@uwf.edu"},
    "History": {"name": "Dr. Brown", "email": "michael.brown@uwf.edu"}
}

UWF_CONFLUENCE_PAGE_SPACE = "public"
UWF_CONFLUENCE_BASE_URL = "https://confluence.uwf.edu"
