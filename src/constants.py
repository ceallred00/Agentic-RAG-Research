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


FAKE_DEPARTMENT_ADVISORS = {
    "Computer Science": {"name": "Dr. Smith", "email": "jane.smith@uwf.edu"}, 
    "Mathematics": {"name": "Dr. Johnson", "email": "john.johnson@uwf.edu"},
    "Biology": {"name": "Dr. Lee", "email": "sarah.lee@uwf.edu"},
    "History": {"name": "Dr. Brown", "email": "michael.brown@uwf.edu"}
}
