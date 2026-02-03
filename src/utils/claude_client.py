from typing import Optional, List, Dict, Any, Union
import threading
import re
import json
from anthropic import Anthropic

from src.utils.chat_api import ChatClient, _select_cfg

class ClaudeClient(ChatClient):
    def __init__(self, provider: Optional[str] = None, config_key: Optional[str] = "claude-3-5-sonnet", api_version: str = None):
        """
        Initialize ClaudeClient.
        Inherits from ChatClient but specializes for Anthropic Claude API.
        """
        if provider is None:
            provider = "claude"
        cfg = _select_cfg(provider, config_key)
        self.model = cfg.get("model") or cfg.get("model_name") or config_key or "claude-3-5-sonnet-20240620"
        self.provider = "claude"
        client_kwargs = {}
        client_kwargs["api_key"] = cfg.get("api_key")
        client_kwargs["base_url"] = "https://api.uniapi.io/claude"
        self.client = Anthropic(**client_kwargs)
        self._lock = threading.Lock()
        cache_id = config_key or self.model or "default"
        safe_cache_id = re.sub(r'[\\/*?:"<>|]', '_', str(cache_id))
        self.cache_file = f"tmp/batch_chat_cache_{safe_cache_id}.jsonl"
        self._cache = {}
        self._load_cache()

    def _convert_openai_tools_to_claude(self, openai_tools: List[Dict]) -> Optional[List[Dict]]:
        """
        Convert OpenAI tool schema to Claude tool schema.
        """
        if not openai_tools:
            return None
        claude_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name")
                description = func.get("description")
                parameters = func.get("parameters")
                claude_tool = {
                    "name": name,
                    "description": description,
                    "input_schema": parameters
                }
                claude_tools.append(claude_tool)
        
        return claude_tools if claude_tools else None

    def _convert_messages_to_claude(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI format messages to Claude format.
        Returns (system_prompt, claude_messages)
        """
        system_prompt = None
        claude_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_prompt = content
            elif role == "user":
                claude_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                parts = []
                reasoning_content = msg.get("reasoning_content")
                if reasoning_content:
                    parts.append({
                        "type": "thinking",
                        "thinking": reasoning_content,
                        "signature": msg.get("thought_signature")
                    })
                
                redacted_thinking_data = msg.get("redacted_thinking_data")
                if redacted_thinking_data:
                    parts.append({
                        "type": "redacted_thinking",
                        "data": redacted_thinking_data
                    })

                if content:
                    parts.append({"type": "text", "text": content})
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        parts.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": func.get("name"),
                            "input": json.loads(func.get("arguments", "{}"))
                        })
                if parts:
                    if len(parts) == 1 and parts[0]["type"] == "text":
                         claude_messages.append({
                            "role": "assistant",
                            "content": parts[0]["text"]
                        })
                    else:
                        claude_messages.append({
                            "role": "assistant",
                            "content": parts
                        })

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id") or msg.get("id") # fallback
                tool_result_content = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(content)
                }
                if claude_messages and claude_messages[-1]["role"] == "user":
                    last_content = claude_messages[-1]["content"]
                    if isinstance(last_content, list):
                        last_content.append(tool_result_content)
                    else:
                        # Convert string to list
                        claude_messages[-1]["content"] = [
                            {"type": "text", "text": last_content},
                            tool_result_content
                        ]
                else:
                    claude_messages.append({
                        "role": "user",
                        "content": [tool_result_content]
                    })
        return system_prompt, claude_messages

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
        thinking_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute chat using Anthropic Claude API.
        Compatible with ChatClient.chat signature.
        """
        if message is None:
            assert prompt is not None, "prompt and message cannot be None at the same time"
            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": prompt})
        else:
            messages = message
        # When thinking is enabled, we must preserve reasoning content in history
        # as Claude API requires assistant messages to start with thinking blocks
        messages = self.filter_reasoning(messages[:])
        system_prompt, claude_messages = self._convert_messages_to_claude(messages)
        
        if system and not system_prompt:
            system_prompt = system
        # 2. Prepare Tools
        tools_config = kwargs.get("tools")
        claude_tools = self._convert_openai_tools_to_claude(tools_config)
        
        # 3. Prepare Request Parameters
        request_params = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": max_tokens or 4096, # Claude requires max_tokens
        }
        if system_prompt:
            request_params["system"] = system_prompt
        if claude_tools:
            request_params["tools"] = claude_tools
        if temperature is not None:
            request_params["temperature"] = temperature
        if top_p is not None:
            request_params["top_p"] = top_p
        if stop is not None:
            request_params["stop_sequences"] = stop

        # 4. Handle Thinking
        if enable_thinking and ("claude-3-7" in self.model or "claude-3-5-sonnet" in self.model or "4-5" in self.model):
             budget = thinking_budget or 8192
             if request_params["max_tokens"] <= budget:
                 request_params["max_tokens"] = budget + 32768 # Ensure ample room for response
             
             request_params["thinking"] = {
                 "type": "enabled",
                 "budget_tokens": budget
             }
             
             request_params["betas"] = ["output-128k-2025-02-19","interleaved-thinking-2025-05-14"]
             request_params.pop("temperature", None)
             request_params.pop("top_p", None)
        try:
            if enable_thinking:
                response = self.client.beta.messages.create(**request_params, timeout=None)
            else:
                response = self.client.messages.create(**request_params)
            content_text, reasoning_content, tool_calls, finish_reason, thought_signature, redacted_thinking_data = "", "", [], response.stop_reason, None, None
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "thinking":
                    reasoning_content += block.thinking
                    thought_signature = block.signature
                elif block.type == "redacted_thinking":
                    redacted_thinking_data = block.data
                elif block.type == "tool_use":
                    # Convert to OpenAI format
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input, ensure_ascii=False)
                        }
                    })
            
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            return {
                "reasoning_content": reasoning_content if reasoning_content else None,
                "content": content_text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "usage": usage,
                "thought_signature": thought_signature,
                "redacted_thinking_data": redacted_thinking_data
            }
        except Exception as e:
            print(f"ClaudeClient Error: {e}")
            raise e
