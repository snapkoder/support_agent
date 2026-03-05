from __future__ import annotations

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROMPTS_DIR = _PROJECT_ROOT / "assets" / "prompts"
KNOWLEDGE_BASE_DIR = _PROJECT_ROOT / "assets" / "knowledge_base"
DATASETS_DIR = _PROJECT_ROOT / "assets" / "datasets"

VAR_DIR = _PROJECT_ROOT / "var"
RAG_INDEX_DIR = VAR_DIR / "rag_index"
LOGS_DIR = VAR_DIR / "logs"
REPORTS_DIR = VAR_DIR / "reports"
