import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DOWNLOAD_DIR = Path.cwd()

DATA_HOME = os.path.join(PROJECT_DIR, "data")

NOTEBOOKS_HOME = os.path.join(PROJECT_DIR, "notebooks")
