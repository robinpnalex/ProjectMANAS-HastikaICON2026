"""Repository defaults; all data entry points also accept explicit paths."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
RUNS_DIR = PROJECT_ROOT / "artifacts/runs"
