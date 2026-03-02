from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Any
from openai import AzureOpenAI, OpenAI
from src.utils.common import read_json_file
from tqdm.auto import tqdm
import json
import re
import os
import hashlib
import time
import threading
import uuid
from src.utils.openai_msg_utils import filter_reasoning

# Suppress tokenizers multi-process warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import fcntl  # Available on non-Windows platforms
except ImportError:
    fcntl = None  # Windows compatibility: do not use fcntl
try:
    from modelscope import AutoTokenizer
    _tokenizer = None  # Lazy loading
except ImportError:
    AutoTokenizer = None
    _tokenizer = None

from pathlib import Path
from src.utils.common import read_config

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def _config_dir() -> Path:
    return _project_root() / "config"

def _load_api_config() -> Dict[str, Any]:
    cfg_dir = _config_dir()
    primary = cfg_dir / "api_key.json"
    fallback = cfg_dir / "api_key_env.json"
    path = primary if primary.exists() else fallback
    return read_json_file(str(path)) if path.exists() else {}
def _select_cfg(provider: Optional[str], config_key: Optional[str]) -> Dict[str, Any]:
    api = _load_api_config()
    if config_key and config_key in api:
        return api[config_key]
    if config_key and config_key not in api and "deepseek_v3.2" in api:
        return api["deepseek_v3.2"]
    if provider == "azure":
        if "gpt-4o" in api:
            return api["gpt-4o"]
        if "gpt" in api:
            return api["gpt"]
    if provider == "vllm" and "vllm" in api:
        return api["vllm"]
    return {}


_chat_client_instance = None
_client_lock = threading.Lock()
_global_cache_lock = threading.Lock()  # Global cache lock to prevent conflict when multiple instances write to the same file

def get_chat_client() -> "ChatClient":
    """
    Get global singleton ChatClient instance
    
    Configuration is read from the chat_model field in config.json:
    {
        "chat_model": {
            "provider": "azure",  // azure, openai, vllm
            "model": "gpt-4o",
            "config_key": "gpt-4o"   // Optional, corresponding to configuration item in api_key.json
        }
    }
    
    Returns:
        ChatClient: Global singleton instance
    """
    global _chat_client_instance
    
    if _chat_client_instance is None:
        with _client_lock:
            if _chat_client_instance is None:
                main_config = read_config()
                chat_config = main_config.get("chat_model", {})
                provider = chat_config.get("provider")
                config_key = chat_config.get("config_key", "deepseek-deepseek-v3.2")
                
                if provider == "gemini":
                    from src.utils.gemini_client import GeminiClient
                    _chat_client_instance = GeminiClient(
                        provider=provider,
                        config_key=config_key
                    )
                elif provider == "claude" or (config_key and "claude" in str(config_key).lower()):
                    from src.utils.claude_client import ClaudeClient
                    _chat_client_instance = ClaudeClient(
                        provider=provider,
                        config_key=config_key
                    )
                else:
                    _chat_client_instance = ChatClient(
                        provider=provider,
                        config_key=config_key
                    )
    
    return _chat_client_instance


class ChatClient:
    def __init__(self, provider: Optional[str] = None, config_key: Optional[str] = "gpt-4o", api_version: str = "2025-01-01-preview"):
        cfg = _select_cfg(provider, config_key)
        # If provider is not specified, try to infer from config
        if provider is None:
            provider = cfg.get("provider", "azure")

        model = cfg.get("model") or cfg.get("model_name")
        if provider == "azure":
            self.client = AzureOpenAI(azure_endpoint=cfg["base_url"], api_key=cfg["api_key"], api_version=api_version)
            self.model = model
            self.provider = "azure"
        elif provider == "vllm":
            self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg.get("api_key", ""))
            self.model = model
            self.provider = "vllm"
        elif provider == "openai":
            self.client = OpenAI(api_key=cfg.get("api_key", ""), base_url=cfg.get("base_url", "https://api.openai.com"))
            self.model = model
            self.provider = "openai"
        else:
            raise ValueError(f"unsupported provider: {provider}")
        
        self._lock = threading.Lock()
        # Use config_key or model to generate an independent cache filename to avoid cache mix-up between different models
        cache_id = config_key or self.model or "default"
        safe_cache_id = re.sub(r'[\\/*?:"<>|]', '_', str(cache_id))
        self.cache_file = f"tmp/batch_chat_cache_{safe_cache_id}.jsonl"
        self._cache = {}  # {cache_hash: result}
        self._load_cache()
    
    @staticmethod
    def _get_tokenizer():
        """Lazy load tokenizer (singleton pattern)"""
        global _tokenizer
        if _tokenizer is None and AutoTokenizer is not None:
            model_path = os.path.join(_project_root(), "models", "Qwen", "Qwen3-Embedding-0.6B")
            _tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
                # cache_dir="./"
            )
        return _tokenizer
    
    @staticmethod
    def count_tokens(
        data: Union[str, List[Dict[str, str]], List[str], List[List[Dict[str, str]]]],
        system: Optional[str] = None
    ) -> Union[int, List[int]]:
        """
        Count the number of tokens
        
        Args:
            data: Supports multiple formats:
                - str: single prompt
                - List[Dict]: single message list
                - List[str]: multiple prompts
                - List[List[Dict]]: multiple messages
            system: System prompt (only used when data is prompt)
            
        Returns:
            Union[int, List[int]]: Returns int for single data, List[int] for batch data
        """
        tokenizer = ChatClient._get_tokenizer()
        if tokenizer is None:
            raise RuntimeError("Tokenizer not installed, please run: pip install modelscope")
        def _count_one(item):
            """Calculate token count for a single item"""
            if isinstance(item, str):
                text = item
                if system:
                    text = f"{system}\n{text}"
                return len(tokenizer.encode(text))
            elif isinstance(item, list):
                if not item:
                    return 0
                return len(tokenizer.apply_chat_template(item, tokenize=True, add_generation_prompt=False))
            return len(tokenizer.encode(str(item)))
        # 1. Determine if it's a single data item
        is_single = False
        if isinstance(data, str):
            is_single = True
            data = [data]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            is_single = True
            data = [data]
        ans = [_count_one(item) for item in data]
        if is_single:
            return ans[0]
        else:
            return ans

    def _parse_stream_response(self, resp) -> Dict[str, Any]:
        reasoning_content = ""
        content = ""
        tool_calls_dict = {}
        finish_reason = None
        usage = None
        for chunk in resp:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage.model_dump()
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if getattr(delta, 'reasoning_content', None):
                reasoning_content += delta.reasoning_content
            if delta.content:
                content += delta.content
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {"name": tool_call.function.name or "", "arguments": ""},
                            "extra_content": getattr(tool_call, 'extra_content', None)
                        }
                    if tool_call.id: tool_calls_dict[idx]["id"] = tool_call.id
                    if tool_call.type: tool_calls_dict[idx]["type"] = tool_call.type
                    if tool_call.function:
                        if tool_call.function.name: tool_calls_dict[idx]["function"]["name"] = tool_call.function.name
                        if tool_call.function.arguments: tool_calls_dict[idx]["function"]["arguments"] += tool_call.function.arguments
                    # Handle streaming extra_content
                    if hasattr(tool_call, 'extra_content') and tool_call.extra_content:
                        if 'extra_content' not in tool_calls_dict[idx]:
                            tool_calls_dict[idx]['extra_content'] = tool_call.extra_content
                        else:
                            tool_calls_dict[idx]['extra_content'] += tool_call.extra_content
        
        return {
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": list(tool_calls_dict.values()) if tool_calls_dict else None,
            "finish_reason": finish_reason,
            "usage": usage or {}
        }

    def _parse_normal_response(self, resp) -> Dict[str, Any]:
        message = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                # Extract extra_content
                extra_content = getattr(tc, 'extra_content', None)
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": args or "{}"},
                    "extra_content": extra_content
                })
        return {
            "reasoning_content": getattr(message, 'reasoning_content', None),
            "content": message.content or "",
            "tool_calls": tool_calls or None,
            "finish_reason": finish_reason
        }

    def chat(
        self, 
        prompt: str = None, 
        message: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = True,
        thinking_budget: Optional[int] = 32000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Single chat call
        
        Args:
            prompt: User-provided prompt (used when message is None)
            message: Complete message list (takes priority)
            system: System prompt (only effective when constructing messages using prompt)
            temperature: Sampling temperature (0-2), higher values are more random
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0-1), works better when choosing between this and temperature
            frequency_penalty: Frequency penalty (-2 to 2), reduces the likelihood of repeating tokens
            presence_penalty: Presence penalty (-2 to 2), encourages new topics
            stop: List of stop sequences
            enable_thinking: Whether to enable thinking mode (applicable to models supporting thinking like Claude/DeepSeek)
            thinking_budget: Thinking token budget (maximum thinking tokens when thinking is enabled)
            **kwargs: Other parameters supported by the API
            
        Returns:
            Dict[str, Any]: Dictionary containing the following fields:
                - reasoning_content: Reasoning/thinking content (if any)
                - content: Actual response content
                - tool_calls: Tool calls (if any)
                - finish_reason: Reason for finishing (stop, length, tool_calls, content_filter, etc.)
        """
        # 1. Build messages
        if message is None:
            assert prompt is not None, "prompt and message cannot both be None"
            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": prompt})
        else:
            messages = message
        import copy
        messages = filter_reasoning(copy.deepcopy(messages), "mimo" not in self.model)
        # 2. Build base request parameters
        request_params = {
            "model": self.model, 
            "messages": messages, 
            "stream": False,
            "temperature": temperature if "gemini" not in self.model else 1.0,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop
        }
        # Filter None values
        request_params = {k: v for k, v in request_params.items() if v is not None}
        if max_tokens is not None:
            key = "max_completion_tokens" if self.provider == "azure" else "max_tokens"
            request_params[key] = max_tokens
            
        # 3. Handle thinking mode
        if enable_thinking:
            if self.provider == "azure" and any(x in self.model.lower() for x in ["o1", "o3"]):
                request_params["reasoning_effort"] = "medium"
            elif ("gpt" in self.model.lower()):
                request_params["reasoning_effort"]= "medium"
            elif self.provider != "azure":
                extra_body = kwargs.pop("extra_body", {})
                extra_body["enable_thinking"] = True
                
                thinking_params = {"type": "enabled"}
                # Claude model requires budget_tokens parameter in thinking dictionary
                if "claude" in self.model.lower() and thinking_budget is not None:
                    thinking_params["budget_tokens"] = thinking_budget
                extra_body["thinking"] = thinking_params
                if thinking_budget is not None:
                    extra_body["thinking_budget"] = thinking_budget
                request_params["extra_body"] = extra_body
                request_params['stream'] = False
        else:
            extra_body = kwargs.pop("extra_body", {})
            thinking_params = {"type": "disabled"}
            extra_body["enable_thinking"] = False
            extra_body["thinking"] = thinking_params
            request_params["extra_body"] = extra_body
            # Explicitly disable thinking mode (for models that support this parameter like Claude)
            if self.provider != "azure" and self.model and "claude" in self.model.lower():
                extra_body = kwargs.pop("extra_body", {})
                extra_body["thinking"] = {"type": "disabled"}
                request_params["extra_body"] = extra_body
        request_params.update(kwargs)
        if request_params.get("tools", None):
            request_params["parallel_tool_calls"] = True
        
        # 4. Execute call
        resp = self.client.chat.completions.create(**request_params)
        # 5. Parse response
        if request_params.get('stream', False):
            result = self._parse_stream_response(resp)
        else:
            result = self._parse_normal_response(resp)
            
        content = result['content']
        reasoning_content = result['reasoning_content']
        tool_calls = result['tool_calls']
        finish_reason = result.get('finish_reason')
        # 6. Fallback extraction strategy
        if enable_thinking and not reasoning_content and content:
            reasoning_content, content = self._extract_reasoning_content(content)
        if not tool_calls and content:
            tool_calls = self._extract_tool_calls(content)
        return {
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else {},
            "messages": messages
        }
    
    def _load_cache(self):
        """Load cache from file to memory"""
        # Load new model-specific cache file + old global cache file
        files_to_load = set([self.cache_file, "tmp/batch_chat_cache.jsonl"])
        
        for file_path in files_to_load:
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            self._cache[entry['hash']] = entry['result']
            except Exception as e:
                print(f"⚠️ Failed to load cache ({file_path}): {e}")
        
        print(f"✓ Loaded cache: {len(self._cache)} records")
    
    def _generate_cache_key(
        self, 
        prompt: Optional[str] = None,
        message: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate cache key (based on all parameters that affect the result)"""
        cache_dict = {
            "provider": self.provider,
            "model": self.model,
            "system": system or "",
            "prompt": prompt or "",
            "message": json.dumps(message, sort_keys=True) if message else "",
            # Only include sampling parameters that affect results
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "max_tokens": kwargs.get("max_tokens"),
            "enable_thinking": kwargs.get("enable_thinking", True),
            "thinking_budget": kwargs.get("thinking_budget"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop": str(kwargs.get("stop")) if kwargs.get("stop") else None,
        }
        # Remove None values
        cache_dict = {k: v for k, v in cache_dict.items() if v is not None}
        cache_str = json.dumps(cache_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    
    def _save_to_cache(
        self,
        cache_key: str,
        result: Dict[str, Any],
        prompt: Optional[str] = None,
        system: Optional[str] = None
    ):
        """Thread-safely save to cache file"""
        # Update memory cache
        self._cache[cache_key] = result
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        # Build cache entry
        entry = {
            "hash": cache_key,
            "key_preview": {  # Fully retained for debugging
                "prompt": prompt or "",
                "system": system or "",
                "model": self.model,
            },
            "result": result,
            "timestamp": int(time.time())
        }
        
        # Append to file (using thread lock for concurrency safety)
        try:
            # Use global lock instead of instance lock to ensure no conflict when multiple instances (e.g., multi-expert evaluation) write to the same file
            with _global_cache_lock:
                with open(self.cache_file, 'a', encoding='utf-8') as f:
                    # Attempt to use fcntl if available (Linux/Mac multi-process safety)
                    if fcntl:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    
                    if fcntl:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"⚠️ Saving cache failed: {e}")

    def _save_batch_script(
        self,
        prompts: Optional[List[str]] = None,
        messages: Optional[List[List[Dict[str, str]]]] = None,
        system: Optional[str] = None,
        script_file: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Save batch requests as JSONL file in OpenAI Batch API format
        
        Args:
            prompts: List of prompts
            messages: List of complete message lists
            system: System prompt
            script_file: Save path, defaults to tmp/batch_script.jsonl
            **kwargs: Other parameters (unused)
            
        Returns:
            Empty list (only saves file, does not call API)
        """
        # Determine save path
        if script_file is None:
            script_file = "tmp/batch_script.jsonl"
        # Ensure directory exists
        os.makedirs(os.path.dirname(script_file), exist_ok=True)
        # Build message list
        all_messages_list = []
        if messages is not None:
            all_messages_list = messages
        else:
            assert prompts is not None, "prompts and messages cannot both be None"
            for prompt in prompts:
                msg_list = []
                if system:
                    msg_list.append({"role": "system", "content": system})
                msg_list.append({"role": "user", "content": prompt})
                all_messages_list.append(msg_list)
        # Write to JSONL file
        with open(script_file, 'w', encoding='utf-8') as f:
            for idx, msg_list in enumerate(all_messages_list):
                entry = {
                    "custom_id": str(idx + 1),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": msg_list
                    }
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"✓ Batch script saved to: {script_file}")
        print(f"  Total {len(all_messages_list)} requests")
        return []

    def batch_chat(
        self, 
        prompts: Optional[List[str]] = None, 
        messages: Optional[List[List[Dict[str, str]]]] = None,
        threads: int = 4, 
        system: Optional[str] = None, 
        enable_thinking: Optional[bool] = False,
        enable_cache: bool = True,  # New: whether to enable cache
        batch_size: int = 10,  # New: batch size
        script: bool = False,  # New: whether to save as batch script format
        script_file: Optional[str] = None,  # New: script file path, defaults to tmp/batch_script.jsonl
        verbose: bool = True,  # New: whether to show progress and logs
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch chat call (supports auto-cache and checkpointing/breakpoint resumption)
        
        Args:
            prompts: List of prompts (used when messages is None)
            messages: List of complete message lists (takes priority)
            threads: Number of concurrent threads
            system: System prompt (only effective when using prompts)
            enable_thinking: Whether to enable thinking mode
            enable_cache: Whether to enable cache (default True, auto-cached to tmp/batch_chat_cache.jsonl)
            batch_size: Batch size (default 10), calls API in batches to avoid long blocking or deadlocks
            script: Whether to save as batch script format (default False), if True doesn't call API, only saves requests to jsonl
            script_file: Script file path (defaults to tmp/batch_script.jsonl)
            verbose: Whether to show progress bar and log info (default True)
            **kwargs: Other sampling parameters passed to the chat method
            
        Returns:
            List[Dict[str, Any]]: List of responses, each containing reasoning_content and content fields, in the same order as input
            If script=True, returns an empty list and saves data to jsonl file
        """
        # If script mode is enabled, save requests in batch script format
        if script:
            return self._save_batch_script(
                prompts=prompts,
                messages=messages,
                system=system,
                script_file=script_file,
                **kwargs
            )
        
        # Prefer using messages, otherwise use prompts
        if messages is not None:
            items = [(i, {"message": msg}) for i, msg in enumerate(messages)]
        else:
            assert prompts is not None, "prompts and messages cannot both be None"
            items = [(i, {"prompt": p, "system": system}) for i, p in enumerate(prompts)]
        
        results = [None] * len(items)
        pending_items = []  # Tasks that actually need to call the LLM
        
        # If cache is enabled, check cache first
        if enable_cache:
            cache_hits = 0
            for i, params in items:
                # Merge parameters to generate cache key
                cache_params = {
                    "enable_thinking": enable_thinking,
                    **params,
                    **kwargs
                }
                # If using prompts mode, system is already in params
                # If using messages mode, it needs to be passed in from outside
                if "system" not in cache_params and system is not None:
                    cache_params["system"] = system
                
                cache_key = self._generate_cache_key(**cache_params)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    cache_hits += 1
                else:
                    pending_items.append((i, params, cache_key))
            if cache_hits > 0 and verbose:
                print(f"✓ Cache hit: {cache_hits}/{len(items)}")
            if pending_items and verbose:
                print(f"⚙ Need to call LLM: {len(pending_items)}/{len(items)}")
        else:
            # Do not use cache, call LLM for all
            pending_items = [(i, params, None) for i, params in items]
        
        # Batch call LLM (only for uncached)
        if pending_items:
            def _run_and_cache(i, params, cache_key):
                result = self.chat(**params, enable_thinking=enable_thinking, **kwargs)
                # If cache is enabled, save immediately
                if enable_cache and cache_key:
                    prompt = params.get("prompt")
                    self._save_to_cache(cache_key, result, prompt=prompt, system=system)
                return i, result
            
            # Batch processing
            total_pending = len(pending_items)
            current_batch_size = batch_size if batch_size > 0 else total_pending
            
            with ThreadPoolExecutor(max_workers=threads) as ex:
                for start_idx in tqdm(range(0, total_pending, current_batch_size), desc="Batch progress", disable=not verbose):
                    end_idx = min(start_idx + current_batch_size, total_pending)
                    batch = pending_items[start_idx:end_idx]
                    futs = [ex.submit(_run_and_cache, i, p, k) for i, p, k in batch]
                    for f in as_completed(futs):
                        i, out = f.result()
                        results[i] = out
        return results
    
    def clear_cache(self, days: Optional[int] = None):
        """
        Clear cache
        
        Args:
            days: Clear cache older than how many days, None means clear all
        """
        if days is None:
            # Clear all
            self._cache.clear()
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            print(f"✓ All cache cleared")
        else:
            # Clear old cache
            cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
            if not os.path.exists(self.cache_file):
                return
            # Read and filter
            temp_file = self.cache_file + ".tmp"
            kept_count = 0
            removed_count = 0
            with open(self.cache_file, 'r', encoding='utf-8') as f_in:
                with open(temp_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        if line.strip():
                            entry = json.loads(line)
                            if entry.get('timestamp', 0) >= cutoff_time:
                                f_out.write(line)
                                kept_count += 1
                            else:
                                removed_count += 1
                                # Remove from memory cache
                                self._cache.pop(entry['hash'], None)
            # Replace original file
            os.replace(temp_file, self.cache_file)
            print(f"✓ Cleared {removed_count} old cache entries (>{days} days), kept {kept_count}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "total_entries": len(self._cache),
            "file_exists": os.path.exists(self.cache_file),
            "file_size_mb": 0,
        }
        if stats["file_exists"]:
            stats["file_size_mb"] = round(os.path.getsize(self.cache_file) / 1024 / 1024, 2)
        return stats
    
    def _extract_reasoning_content(self, content: str) -> tuple[Optional[str], str]:
        """
        Extract reasoning content from multiple formats.
        Returns (reasoning_content, remaining_content).
        Supported formats:
        - <think>...</think>
        - <reasoning>...</reasoning>
        - <thought>...</thought>
        """
        if not content:
            return None, content
        # List of tags to check in priority order
        tags = ['think', 'reasoning', 'thought']
        extracted_parts = []
        remaining_content = content
        for tag in tags:
            pattern = re.compile(f'<{tag}>(.*?)</{tag}>', re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(remaining_content)
            if matches:
                # Add found matches
                extracted_parts.extend([m.strip() for m in matches])
                # Remove tags from content
                remaining_content = pattern.sub('', remaining_content).strip()
        if extracted_parts:
            return "\n\n".join(extracted_parts), remaining_content
        return None, content

    def _extract_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tool calls from multiple formats.
        Supported formats:
        1. JSON in <tool_call> tag:
           <tool_call>{"tool": "name", "params": {...}}</tool_call>
           <tool_call>{"name": "name", "arguments": {...}}</tool_call>
           
        2. XML function calls:
           <function_calls>
               <invoke name="get_weather">
                   <parameter name="location">Beijing</parameter>
               </invoke>
           </function_calls>
        """
        if not content:
            return None
        tool_calls = []
        # 1. Try JSON format <tool_call>
        json_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL | re.IGNORECASE)
        json_matches = json_pattern.findall(content)
        for match in json_matches:
            try:
                match = match.strip()
                try:
                    data = json.loads(match)
                except json.JSONDecodeError:
                    data, _ = json.JSONDecoder().raw_decode(match)
                if isinstance(data, dict):
                    # Handle different JSON structures
                    name = data.get("tool") or data.get("name") or data.get("function")
                    params = data.get("params") or data.get("arguments") or data.get("parameters") or {}
                    
                    if name:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(params, ensure_ascii=False) if isinstance(params, dict) else str(params)
                            }
                        })
            except Exception:
                pass
                
        # 2. Try XML format <function_calls> or <tool_code>
        # Simple XML parser for <invoke name="...">...</invoke> structure
        xml_pattern = re.compile(r'<invoke\s+name=["\'](.*?)["\']\s*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE)
        xml_matches = xml_pattern.findall(content)
        
        for name, body in xml_matches:
            try:
                params = {}
                # Extract parameters: <parameter name="key">value</parameter>
                param_pattern = re.compile(r'<parameter\s+name=["\'](.*?)["\']\s*>(.*?)</parameter>', re.DOTALL | re.IGNORECASE)
                param_matches = param_pattern.findall(body)
                for p_name, p_value in param_matches:
                    params[p_name] = p_value.strip()
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": params
                    }
                })
            except Exception:
                pass

        # 3. Try Markdown JSON block
        if not tool_calls:
             code_block_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)
             blocks = code_block_pattern.findall(content)
             for block in blocks:
                 try:
                    block = block.strip()
                    try:
                        data = json.loads(block)
                    except json.JSONDecodeError:
                        data, _ = json.JSONDecoder().raw_decode(block)
                    
                    if isinstance(data, dict):
                        name = data.get("tool") or data.get("name") or data.get("function")
                        params = data.get("params") or data.get("arguments") or data.get("parameters") or {}
                        
                        if name:
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(params, ensure_ascii=False) if isinstance(params, dict) else str(params)
                                }
                            })
                 except Exception:
                     pass

        return tool_calls if tool_calls else None

