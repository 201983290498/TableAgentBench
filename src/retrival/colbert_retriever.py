"""
ColBERT Retriever - Late Interaction retrieval based on ragatouille
"""
from typing import List, Dict, Any
import os

from src.retrival.base import SimilarityResult, MODEL_CACHE_DIR


class ColBERTRetriever:
    """
    ColBERT Retriever - Late Interaction retrieval based on ragatouille
    
    ColBERT uses token-level interaction, and the retrieval effect is usually better than ordinary embeddings,
    but the computational overhead is larger, suitable for scenarios with high accuracy requirements.
    
    Example:
        from src.retrival import ColBERTRetriever
        
        retriever = ColBERTRetriever()
        
        # Add documents and build index
        retriever.index(["Document 1 content", "Document 2 content"], index_name="my_index")
        
        # Search
        results = retriever.search("query content", top_k=5)
        
        # With rerank
        results = retriever.search("query", top_k=10, rerank=True, rerank_top_k=3)
    """
    
    def __init__(
        self, 
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = None
    ):
        """
        Args:
            model_name: ColBERT model name or local path
            device: Execution device
        """
        self.model_name = model_name
        self._device = device
        self._model = None
        self._index_name = None
        self._corpus: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
    
    def _init_model(self):
        if self._model is not None:
            return
        
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            raise ImportError("ragatouille not installed, please run: pip install ragatouille")
        
        # Check local path or use model name
        if os.path.exists(self.model_name):
            model_path = self.model_name
        else:
            # Try to load from cache
            cache_path = os.path.join(MODEL_CACHE_DIR, self.model_name.replace("/", os.sep)) if MODEL_CACHE_DIR else None
            model_path = cache_path if cache_path and os.path.exists(cache_path) else self.model_name
        
        self._model = RAGPretrainedModel.from_pretrained(model_path)
    
    def index(
        self, 
        documents: List[str], 
        index_name: str = "colbert_index",
        metadata: List[Dict[str, Any]] = None,
        split_documents: bool = False
    ) -> None:
        """
        Build ColBERT index
        
        Args:
            documents: List of documents
            index_name: Index name
            metadata: Document metadata
            split_documents: Whether to automatically split long documents
        """
        self._init_model()
        
        self._corpus = documents
        self._metadata = metadata or [{} for _ in documents]
        self._index_name = index_name
        
        self._model.index(
            index_name=index_name,
            collection=documents,
            split_documents=split_documents
        )
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        rerank: bool = False,
        rerank_top_k: int = None
    ) -> List[SimilarityResult]:
        """
        Search
        
        Args:
            query: Query text
            top_k: Number of results to return
            rerank: Whether to perform rerank
            rerank_top_k: Number of results to keep after rerank
        """
        self._init_model()
        
        if self._index_name is None:
            raise ValueError("Please call index() to build the index first")
        
        results = self._model.search(
            query, 
            index_name=self._index_name, 
            k=top_k,
            force_fast=False
        )
        
        if rerank and results:
            documents = [r['content'] for r in results]
            rerank_k = rerank_top_k or top_k
            results = self._model.rerank(query=query, documents=documents, k=rerank_k)
        
        # Convert to uniform format
        output = []
        for i, r in enumerate(results):
            # Find original index
            content = r.get('content', '')
            try:
                idx = self._corpus.index(content)
                meta = self._metadata[idx]
            except ValueError:
                idx = i
                meta = {}
            
            output.append(SimilarityResult(
                index=idx,
                text=content,
                score=float(r.get('score', 0)),
                metadata=meta
            ))
        
        return output
    
    def search_batch(
        self, 
        queries: List[str], 
        top_k: int = 5,
        rerank: bool = False,
        rerank_top_k: int = None
    ) -> List[List[SimilarityResult]]:
        """Batch search"""
        return [self.search(q, top_k, rerank, rerank_top_k) for q in queries]
