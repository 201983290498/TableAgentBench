"""
Faiss Vector Index - For large-scale semantic retrieval
"""
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

from src.retrival.base import SimilarityResult, _check_faiss_available

if TYPE_CHECKING:
    from src.retrival.embedder import Embedder


class FaissIndex:
    """
    Faiss Vector Index - For large-scale semantic retrieval
    
    Example:
        from src.retrival import Embedder, FaissIndex
        
        embedder = Embedder("bge-small-zh-v1.5")
        index = FaissIndex(embedder)
        
        # Build index
        index.add(["Document 1", "Document 2", "Document 3"])
        
        # Search
        results = index.search("query", top_k=2)
        
        # Save/load
        index.save("index.faiss")
        index.load("index.faiss")
    """
    
    def __init__(
        self, 
        embedder: 'Embedder' = None,
        use_gpu: bool = False,
        index_type: str = "flat"  # "flat" | "ivf" | "hnsw"
    ):
        """
        Args:
            embedder: Embedder instance, uses default if None
            use_gpu: Whether to use GPU acceleration
            index_type: Index type flat(accurate) | ivf(approximate) | hnsw(approximate)
        """
        if not _check_faiss_available():
            raise ImportError("faiss not installed, please run: pip install faiss-cpu or faiss-gpu")
        
        import faiss
        self._faiss = faiss
        
        self._embedder = embedder
        self._use_gpu = use_gpu
        self._index_type = index_type
        self._index = None
        self._corpus: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
    
    @property
    def embedder(self) -> 'Embedder':
        if self._embedder is None:
            from src.retrival.embedder import get_default_embedder
            self._embedder = get_default_embedder()
        return self._embedder
    
    @property
    def dimension(self) -> int:
        return self.embedder.dimension
    
    @property
    def size(self) -> int:
        return len(self._corpus)
    
    def _create_index(self, dimension: int, n_vectors: int = 0):
        """Create faiss index"""
        if self._index_type == "flat":
            index = self._faiss.IndexFlatIP(dimension)  # Inner Product (equal to cosine similarity after normalization)
        elif self._index_type == "ivf":
            nlist = max(1, min(n_vectors // 10, 100))
            quantizer = self._faiss.IndexFlatIP(dimension)
            index = self._faiss.IndexIVFFlat(quantizer, dimension, nlist, self._faiss.METRIC_INNER_PRODUCT)
        elif self._index_type == "hnsw":
            index = self._faiss.IndexHNSWFlat(dimension, 32, self._faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self._index_type}")
        
        if self._use_gpu:
            res = self._faiss.StandardGpuResources()
            index = self._faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def add(
        self, 
        texts: List[str], 
        metadata: List[Dict[str, Any]] = None,
        batch_size: int = 32
    ) -> None:
        """
        Add documents to index
        
        Args:
            texts: List of documents
            metadata: Metadata for each document
            batch_size: Encoding batch size
        """
        if not texts:
            return
        
        # Encoding
        embeddings = self.embedder.encode_batch(texts, batch_size=batch_size)
                
        # Normalization (for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = (embeddings / norms).astype(np.float32)
        
        # Create or update index
        if self._index is None:
            self._index = self._create_index(self.dimension, len(texts))
            if self._index_type == "ivf":
                self._index.train(embeddings)
        
        self._index.add(embeddings)
        self._corpus.extend(texts)
        
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])
    
    def search(self, query: str, top_k: int = 5) -> List[SimilarityResult]:
        """
        Search
        
        Args:
            query: Query text
            top_k: Number of results to return
        """
        if self._index is None or self.size == 0:
            return []
        
        # Encode query
        query_emb = self.embedder.encode(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.astype(np.float32).reshape(1, -1)
        
        # Search
        top_k = min(top_k, self.size)
        scores, indices = self._index.search(query_emb, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # faiss returns -1 for invalid
                continue
            results.append(SimilarityResult(
                index=int(idx),
                text=self._corpus[idx],
                score=float(score),
                metadata=self._metadata[idx]
            ))
        
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 5) -> List[List[SimilarityResult]]:
        """Batch search"""
        if self._index is None or self.size == 0:
            return [[] for _ in queries]
        
        query_embs = self.embedder.encode_batch(queries, show_progress=False)
        norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        query_embs = (query_embs / norms).astype(np.float32)
        
        top_k = min(top_k, self.size)
        scores_batch, indices_batch = self._index.search(query_embs, top_k)
        
        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                results.append(SimilarityResult(
                    index=int(idx),
                    text=self._corpus[idx],
                    score=float(score),
                    metadata=self._metadata[idx]
                ))
            all_results.append(results)
        
        return all_results
    
    def clear(self) -> None:
        """Clear index"""
        self._index = None
        self._corpus = []
        self._metadata = []
    
    def save(self, path: str) -> None:
        """Save index to file"""
        import pickle
        
        # Save faiss index
        index_path = path if path.endswith('.faiss') else path + '.faiss'
        if self._use_gpu:
            cpu_index = self._faiss.index_gpu_to_cpu(self._index)
            self._faiss.write_index(cpu_index, index_path)
        else:
            self._faiss.write_index(self._index, index_path)
        
        # Save metadata
        meta_path = index_path.replace('.faiss', '.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'corpus': self._corpus,
                'metadata': self._metadata,
                'index_type': self._index_type
            }, f)
    
    def load(self, path: str) -> None:
        """Load index from file"""
        import pickle
        
        index_path = path if path.endswith('.faiss') else path + '.faiss'
        self._index = self._faiss.read_index(index_path)
        
        if self._use_gpu:
            res = self._faiss.StandardGpuResources()
            self._index = self._faiss.index_cpu_to_gpu(res, 0, self._index)
        
        meta_path = index_path.replace('.faiss', '.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            self._corpus = meta['corpus']
            self._metadata = meta['metadata']
            self._index_type = meta.get('index_type', 'flat')
