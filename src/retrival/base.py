"""
Base definition for retrieval module
"""
from typing import Dict, Any
from dataclasses import dataclass, field
import os

from src.utils.common import get_model_cache_dir


# Model cache directory
_dir = get_model_cache_dir() or os.environ.get("HF_HOME")
if _dir:
    os.environ["HF_HOME"] = _dir
    os.environ["TRANSFORMERS_CACHE"] = _dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = _dir
    os.environ["MODELSCOPE_CACHE"] = _dir
MODEL_CACHE_DIR = _dir


@dataclass
class SimilarityResult:
    """Similarity retrieval result"""
    index: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _check_faiss_available() -> bool:
    """Check if faiss is available"""
    try:
        import faiss
        return True
    except ImportError:
        return False
