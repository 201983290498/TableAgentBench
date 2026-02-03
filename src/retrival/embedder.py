"""
Embedding utility class - Supports Encoder and Decoder type embedding models
"""
from typing import List, Optional, Literal, Any
import numpy as np
import os
import torch
from modelscope import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.utils.common import get_model_cache_dir, get_default_embedding_model
from src.retrival.base import SimilarityResult
import pickle
import gc
import torch.nn.functional as F
try:
    import torch_npu
except ImportError:
    pass

# Model configuration: model name -> (type, dimension)
MODEL_CONFIGS = {
    # Encoder type - sentence-transformers
    "paraphrase-multilingual-MiniLM-L12-v2": ("encoder", 384),
    "all-MiniLM-L6-v2": ("encoder", 384),
    "bge-small-zh-v1.5": ("encoder", 512),
    "bge-base-zh-v1.5": ("encoder", 768),
    "bge-large-zh-v1.5": ("encoder", 1024),
    # Decoder type - uses last token pooling
    "./models/Qwen/Qwen3-Embedding-0.6B": ("decoder", 1024),
    "Qwen/Qwen3-Embedding-0.6B": ("decoder", 1024),
    "Qwen/Qwen3-Embedding-4B": ("decoder", 2560),
    "Qwen/Qwen3-Embedding-8B": ("decoder", 4096),
    # TF-IDF
    "tfidf": ("tfidf", 512),
}


class Embedder:
    """
    Unified Embedding class
    
    Example:
        # Encoder type (sentence-transformers)
        embedder = Embedder("bge-small-zh-v1.5")
        
        # Decoder type (Qwen3-Embedding, last token pooling)
        embedder = Embedder("Qwen/Qwen3-Embedding-0.6B")
        
        # Retrieval (default memory calculation)
        results = embedder.retrieve("futures data", ["futures.csv", "stocks.csv"], top_k=2)
        
        # Accelerate with Faiss (recommended for large-scale data)
        embedder = Embedder("bge-small-zh-v1.5", use_faiss=True)
    """
    
    def __init__(
        self, 
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        type: Literal["encoder", "decoder", "tfidf"] = None,
        dimension: int = None,
        device: str = None,
        max_length: int = 8192,
        use_faiss: bool = False,
        **kwargs
    ):
        """
        Args:
            model: Model name
            type: Model type "encoder" | "decoder" | "tfidf"
            dimension: Embedding dimension
            device: Run device "cuda" | "cpu"
            max_length: Decoder model maximum length
            use_faiss: Whether to use Faiss for vector retrieval (recommended for large-scale data)
        """
        self.model_name = model
        self.max_length = max_length
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        if device:
            self._device = device
        else:
            if torch.backends.mps.is_available():  # ✅ Prefer M4 GPU
                self._device = "mps"
            elif torch.cuda.is_available():        # ✅ Backup NVIDIA GPU
                self._device = "cuda"
            elif hasattr(torch, 'npu') and torch.npu.is_available(): # ✅ NPU
                self._device = "npu"
            else:                                  # ✅ Finally use CPU
                self._device = "cpu"    
        self.use_faiss = use_faiss
        self._faiss_index = None  # Lazy loading
        self._remote_client = None
        
        assert model in MODEL_CONFIGS or type is not None, "Unknown model, please specify type parameter"
        # Get from configuration or use specified value
        if model in MODEL_CONFIGS:
            config_type, config_dim = MODEL_CONFIGS[model]
            self.type = type or config_type
            self.dimension = dimension or config_dim
            
        if self._remote_client is None:
            self._init_model()
        
    def _init_model(self):
        if self._model is not None:
            return
        if self.type == "encoder":
            self._model = SentenceTransformer(self.model_name, device=self._device)
            self.dimension = self._model.get_sentence_embedding_dimension()
            
        elif self.type == "decoder":
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
            self._model.eval()
            
        elif self.type == "tfidf":
            self._model = TfidfVectorizer(max_features=self.dimension)
            self._fitted = False

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Last token pooling for Decoder models"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text, returning a (dimension,) vector"""
        return self.encode_batch([text], show_progress=True)[0]
    
    def encode_batch(self, texts: List[str], batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
        """
        Batch encoding, returning an (n, dimension) matrix
        
        Args:
            texts: List of texts
            batch_size: Number of texts processed per batch
            show_progress: Whether to show progress bar
        """
        from tqdm import tqdm
        self._init_model()
        if not texts:
            return np.array([])
        if self.type == "encoder":
            # sentence-transformers already performs internal batch processing
            return self._model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=show_progress, 
                convert_to_numpy=True
            )
            
        elif self.type == "decoder":
            result_embeddings = None
            iterator = range(0, len(texts), batch_size)
            if show_progress and len(texts) > batch_size:
                iterator = tqdm(iterator, desc="Encoding", unit="batch")
            with torch.no_grad():
                for i in iterator:
                    batch_texts = texts[i:i + batch_size]
                    batch_dict = self._tokenizer(
                        batch_texts, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors="pt"
                    ).to(self._device)
                    outputs = self._model(**batch_dict, use_cache=False)
                    embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    batch_emb_numpy = embeddings.cpu().numpy()
                    if result_embeddings is None:
                        dim = batch_emb_numpy.shape[1]
                        result_embeddings = np.zeros((len(texts), dim), dtype=np.float32)
                    end_idx = min(i + batch_size, len(texts))
                    result_embeddings[i:end_idx] = batch_emb_numpy
                    del outputs, embeddings, batch_dict
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    if hasattr(torch, 'mps') and torch.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
            return result_embeddings if result_embeddings is not None else np.array([])
            
        elif self.type == "tfidf":
            # TF-IDF: fit on the first call, transform thereafter
            if not self._fitted:
                self._model.fit(texts)
                self._fitted = True
            return self._model.transform(texts).toarray()

    def retrieve(self, query: str, corpus: List[str], top_k: int = 5, cache_path: str = None, batch_size: int = 8) -> List[SimilarityResult]:
        """Semantic retrieval, returning results sorted by similarity in descending order"""
        # if getattr(self, '_remote_client', None):
        #      return self._remote_client.retrieve(query, corpus, top_k=top_k, cache_path=cache_path)

        if not corpus:
            return []
        if self.use_faiss:
            return self._retrieve_faiss(query, corpus, top_k, cache_path, batch_size)
        else:
            return self._retrieve_numpy(query, corpus, top_k, cache_path, batch_size)
    
    def _retrieve_numpy(self, query: str, corpus: List[str], top_k: int, cache_path: str = None, batch_size: int = 8) -> List[SimilarityResult]:
        """Use numpy for in-memory calculation"""
        # 1. Try to load cache
        cached_data = {} # text -> embedding
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict) and 'texts' in loaded and 'embeddings' in loaded:
                    if len(loaded['texts']) == len(loaded['embeddings']):
                        cached_data = {t: e for t, e in zip(loaded['texts'], loaded['embeddings'])}
            except Exception:
                pass
        # 2. Check which ones need calculation, use dictionary to deduplicate while maintaining order (Python 3.7+ dict is ordered, but here mainly for lookup)
        text_to_emb, missing_texts, seen_missing = {}, [], set()
        for text in corpus:
            if text in cached_data:
                text_to_emb[text] = cached_data[text]
            else:
                if text not in seen_missing:
                    missing_texts.append(text)
                    seen_missing.add(text)
        # 3. Calculate missing embeddings
        if missing_texts:
            # print(f"Encoding {len(missing_texts)} missing texts...")
            new_embs = self.encode_batch(missing_texts, batch_size=batch_size, show_progress=True)
            # Update mapping for current task and global cache
            for text, emb in zip(missing_texts, new_embs):
                text_to_emb[text] = emb
                cached_data[text] = emb
            # 4. Save updated cache (incremental update)
            if cache_path:
                try:
                    cache_dir = os.path.dirname(cache_path)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    # Convert cache to list format to save
                    save_texts = list(cached_data.keys())
                    save_embs = np.array([cached_data[t] for t in save_texts])
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'texts': save_texts, 'embeddings': save_embs}, f)
                except Exception as e:
                    print(f"Warning: Failed to save cache to {cache_path}: {e}")
        # 5. Assemble final embedding matrix (strictly in corpus order)
        corpus_emb = np.array([text_to_emb[text] for text in corpus])
        query_emb = self._model.transform([query]).toarray()[0] if self.type == "tfidf" else self.encode(query)
        # Cosine similarity (decoder is already normalized, direct dot product)
        scores = np.dot(corpus_emb, query_emb)
        top_k = min(top_k, len(corpus))
        if top_k <= 0:
            return []
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [SimilarityResult(int(i), corpus[i], float(scores[i])) for i in top_idx]
    
    def _retrieve_faiss(self, query: str, corpus: List[str], top_k: int, cache_path: str = None, batch_size: int = 8) -> List[SimilarityResult]:
        """Use Faiss for vector retrieval"""
        from src.retrival.faiss_index import FaissIndex
        
        # Lazy load or rebuild index
        if self._faiss_index is None:
            self._faiss_index = FaissIndex(embedder=self)
        
        # Clear and rebuild index
        self._faiss_index.clear()
        self._faiss_index.add(corpus, batch_size=batch_size)
        
        return self._faiss_index.search(query, top_k)


# Global singleton
_default_embedder: Optional[Embedder] = None
_use_remote_client: bool = False  # Whether to use remote Client mode


def enable_remote_client_mode():
    """Enable remote Client mode (call in sub-processes)"""
    global _use_remote_client
    _use_remote_client = True


def disable_remote_client_mode():
    """Disable remote Client mode"""
    global _use_remote_client
    _use_remote_client = False


def is_remote_client_mode():
    """Check if in remote Client mode"""
    global _use_remote_client
    return _use_remote_client


def get_default_embedder(model: str = None, **kwargs) -> Embedder:
    """
    Get the default Embedder instance (lazy loading)
    
    If in remote Client mode, returns EmbeddingClient;
    otherwise returns a local Embedder instance.
    """
    global _default_embedder, _use_remote_client
    
    # If in remote mode, return Client
    if _use_remote_client:
        from src.retrival.embedder_service import EmbeddingServiceManager
        if EmbeddingServiceManager.is_service_mode():
            return EmbeddingServiceManager.get_client()
    
    # Otherwise use local Embedder
    if _default_embedder is None:
        if model is None:
            model = get_default_embedding_model() or "tfidf"
        _default_embedder = Embedder(model, **kwargs)
    return _default_embedder


def semantic_search(query: str, corpus: List[str], top_k: int = 5, embedder: Embedder = None, cache_path: str = None, batch_size: int = 8) -> List[SimilarityResult]:
    """Convenient semantic search"""
    return (embedder or get_default_embedder()).retrieve(query, corpus, top_k, cache_path, batch_size=batch_size)