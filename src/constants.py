from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

env_root = os.getenv("PROJECT_ROOT", "./")
# Wrap env_root in Path for easier path manipulations
ROOT_DIR = Path(env_root)
CONFIGS_DIR = ROOT_DIR / "configs"
SRC_DIR = ROOT_DIR / "src"
LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "app.log"

BASE_DIAGRAM_DIR = ROOT_DIR / "diagrams"
PROD_DIAGRAM_DIR = BASE_DIAGRAM_DIR / "production"
