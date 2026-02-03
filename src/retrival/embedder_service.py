
import multiprocessing
from multiprocessing import Manager
import torch
import time
import queue
from typing import List
from src.retrival.embedder import Embedder, SimilarityResult
from src.retrival.base import SimilarityResult
from src.utils.common import get_default_device
# Define the protocol for requests and responses
class EmbeddingRequest:
    def __init__(self, request_id, method, args, kwargs):
        self.request_id = request_id
        self.method = method # "encode", "retrieve"
        self.args = args
        self.kwargs = kwargs

class EmbeddingResponse:
    def __init__(self, request_id, result=None, error=None):
        self.request_id = request_id
        self.result = result
        self.error = error

class EmbeddingServer:
    """
    Embedding service running in an independent process or thread.
    Responsible for loading the model and serially processing requests from Clients.
    """
    def __init__(self, model_name: str = None, device: str = None, request_queue: multiprocessing.Queue = None, response_queue: multiprocessing.Queue = None):
        self.device = device or get_default_device()
        self.model_name = model_name
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.embedder = None
        self._running = True
        

    def run(self):
        print(f"[EmbeddingServer] Initializing model '{self.model_name}' on {self.device}...")
        try:
            # Actually load the model (only load once here)
            self.embedder = Embedder(model=self.model_name, device=self.device)
            print("[EmbeddingServer] Model loaded successfully.")
        except Exception as e:
            print(f"[EmbeddingServer] Failed to load model: {e}")
            return

        while self._running:
            try:
                # Block and wait for request
                req: EmbeddingRequest = self.request_queue.get(timeout=1)
                if req is None: # Stop signal
                    break
                
                self._process_request(req)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[EmbeddingServer] Error loop: {e}")

    def _process_request(self, req: EmbeddingRequest):
        try:
            result = None
            if req.method == "encode":
                # text: str
                result = self.embedder.encode(*req.args, **req.kwargs)
            elif req.method == "encode_batch":
                # texts: List[str]
                result = self.embedder.encode_batch(*req.args, **req.kwargs)
            elif req.method == "retrieve":
                # query, corpus, top_k
                result = self.embedder.retrieve(*req.args, **req.kwargs)
            else:
                raise ValueError(f"Unknown method: {req.method}")
            
            resp = EmbeddingResponse(req.request_id, result=result)
            self.response_queue.put(resp)
        except Exception as e:
            self.response_queue.put(EmbeddingResponse(req.request_id, error=e))

class EmbeddingClient:
    """
    Lightweight client that replaces the original Embedder instance.
    All method calls will be forwarded to the Server.
    """
    def __init__(self, request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        # Does not need to load the model, consumes almost no VRAM

    def encode(self, text: str) -> any:
        return self._call_remote("encode", text)

    def encode_batch(self, texts: List[str], **kwargs) -> any:
        return self._call_remote("encode_batch", texts, **kwargs)

    def retrieve(self, query: str, corpus: List[str], top_k: int = 5, cache_path: str = None, batch_size: int = 8) -> List[SimilarityResult]:
        return self._call_remote("retrieve", query, corpus, top_k=top_k, cache_path=cache_path, batch_size=batch_size)

    def _call_remote(self, method, *args, **kwargs):
        import uuid
        req_id = str(uuid.uuid4())
        req = EmbeddingRequest(req_id, method, args, kwargs)
        
        # Send request
        self.request_queue.put(req)
        while True:
            resp: EmbeddingResponse = self.response_queue.get()
            if resp.request_id == req_id:
                if resp.error:
                    raise resp.error
                return resp.result
            else:
                self.response_queue.put(resp) # Put it back
                time.sleep(0.01)

# -----------------------------------------------------------------------------
# Service Manager (Helper to start server)
# -----------------------------------------------------------------------------

# Global Manager instance, used to create Queues shared across processes
_manager = None
_shared_req_queue = None
_shared_resp_queue = None


def _get_manager():
    """Get the global Manager instance"""
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


def get_shared_queues():
    """Get shared queues for use in child processes"""
    global _shared_req_queue, _shared_resp_queue
    return _shared_req_queue, _shared_resp_queue


def set_shared_queues(req_queue, resp_queue):
    """Set shared queues (called during child process initialization)"""
    global _shared_req_queue, _shared_resp_queue
    _shared_req_queue = req_queue
    _shared_resp_queue = resp_queue


class EmbeddingServiceManager:
    _instance = None
    _server_process = None
    _req_queue = None
    _resp_queue = None
    _is_service_mode = False  # Flag to indicate if using service mode

    @classmethod
    def start_service(cls, model_name=None, device=None):
        """Start Embedding service (called by main process)"""
        global _shared_req_queue, _shared_resp_queue
        
        if cls._server_process is not None and cls._server_process.is_alive():
            return cls._req_queue, cls._resp_queue
        device = device or get_default_device()
        # Use Manager to create queues that can be shared across processes
        manager = _get_manager()
        cls._req_queue = manager.Queue()
        cls._resp_queue = manager.Queue()
        
        # Save to global variables for use by get_shared_queues
        _shared_req_queue = cls._req_queue
        _shared_resp_queue = cls._resp_queue
        
        server = EmbeddingServer(model_name, device, cls._req_queue, cls._resp_queue)
        cls._server_process = multiprocessing.Process(target=server.run, daemon=True)
        cls._server_process.start()
        cls._is_service_mode = True
        print(f"[ServiceManager] Service started (PID: {cls._server_process.pid})")
        
        return cls._req_queue, cls._resp_queue

    @classmethod
    def get_client(cls):
        """Get Client instance"""
        global _shared_req_queue, _shared_resp_queue
        
        # Prioritize using global shared queues (child process scenario)
        req_q = _shared_req_queue or cls._req_queue
        resp_q = _shared_resp_queue or cls._resp_queue
        
        if req_q is None:
            raise RuntimeError("Service not started. Call start_service() first or set_shared_queues().")
        return EmbeddingClient(req_q, resp_q)

    @classmethod
    def is_service_mode(cls):
        """Check if in service mode"""
        global _shared_req_queue
        return cls._is_service_mode or _shared_req_queue is not None

    @classmethod
    def stop_service(cls):
        if cls._req_queue:
            cls._req_queue.put(None)
        if cls._server_process:
            cls._server_process.join()
        cls._is_service_mode = False
