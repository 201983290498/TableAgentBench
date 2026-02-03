"""
Retrieval Module

Provides:
- Embedder: Unified embedding encoder
- FaissIndex: Faiss vector index
- ColBERTRetriever: ColBERT retriever
- semantic_search: Convenient semantic search function

Example:
    # Basic semantic search
    from src.retrival import semantic_search
    results = semantic_search("query", ["doc1", "doc2"], top_k=2)
    
    # Using Embedder
    from src.retrival import Embedder
    embedder = Embedder("bge-small-zh-v1.5")
    results = embedder.retrieve("query", corpus, top_k=5)
    
    # Using Faiss for large-scale retrieval
    from src.retrival import Embedder, FaissIndex
    index = FaissIndex(Embedder("bge-small-zh-v1.5"))
    index.add(["doc1", "doc2", ...])
    results = index.search("query", top_k=5)
    
    # Using ColBERT
    from src.retrival import ColBERTRetriever
    retriever = ColBERTRetriever()
    retriever.index(["doc1", "doc2"])
    results = retriever.search("query", top_k=5)
"""

from src.retrival.base import SimilarityResult, MODEL_CACHE_DIR
from src.retrival.embedder import (
    Embedder,
    MODEL_CONFIGS,
    get_default_embedder,
    semantic_search,
)
from src.retrival.faiss_index import FaissIndex
from src.retrival.colbert_retriever import ColBERTRetriever

__all__ = [
    # Base
    "SimilarityResult",
    "MODEL_CACHE_DIR",
    # Embedder
    "Embedder",
    "MODEL_CONFIGS",
    "get_default_embedder",
    "semantic_search",
    # Faiss
    "FaissIndex",
    # ColBERT
    "ColBERTRetriever",
]
