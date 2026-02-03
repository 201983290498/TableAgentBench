from typing import Any, Optional
from src.utils.chat_api import ChatClient, get_chat_client


class BaseLLM:
    
    def __init__(self, client: Optional[ChatClient] = None, system_prompt: str = None):
        # Prioritize the passed client, otherwise use the singleton
        self.client = client if client is not None else get_chat_client()
        assert system_prompt is not None, "system_prompt cannot be empty"
        self.system_prompt = system_prompt
        
    def __call__(self) -> Any:
        raise NotImplementedError("Subclasses must implement the __call__ method")
