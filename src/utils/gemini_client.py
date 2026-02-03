from typing import Optional, List, Dict, Any
import threading
import re
import json
import uuid
from google import genai
from google.genai import types

from src.utils.chat_api import ChatClient, _select_cfg

class GeminiClient(ChatClient):
    def __init__(self, provider: Optional[str] = None, config_key: Optional[str] = "gpt-4o", api_version: str = "2025-01-01-preview"):
        """
        Initialize GeminiClient.
        Inherits from ChatClient but specializes for Google Gemini API.
        """
        # 1. Load configuration
        cfg = _select_cfg(provider, config_key)
        if provider is None:
            provider = cfg.get("provider", "gemini")
        self.model = cfg.get("model") or cfg.get("model_name")
        self.provider = "gemini"
        
        # 2. Initialize Gemini Client
        api_key = cfg.get("api_key")
        self.base_url = 'https://api.uniapi.io/gemini'
        http_options = types.HttpOptions(base_url=self.base_url)
        self.client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )
        
        # 3. Initialize cache mechanism (Replicating ChatClient logic)
        self._lock = threading.Lock()
        cache_id = config_key or self.model or "default"
        safe_cache_id = re.sub(r'[\\/*?:"<>|]', '_', str(cache_id))
        self.cache_file = f"tmp/batch_chat_cache_{safe_cache_id}.jsonl"
        self._cache = {}
        self._load_cache()

    def _convert_openai_tools_to_gemini(self, openai_tools: List[Dict]) -> Optional[List[types.Tool]]:
        """
        Convert OpenAI tool schema to Gemini tool schema.
        """
        if not openai_tools:
            return None
        
        declarations = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name")
                description = func.get("description")
                parameters = func.get("parameters")
                declarations.append(types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters 
                ))
        if not declarations:
            return None
        return [types.Tool(function_declarations=declarations)]

    def _convert_message_to_gemini(self, messages):
        contents, system_instruction = [], None
        tool_call_id_map = {}
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))
            elif role == "assistant":
                parts = []
                # Add thought part if available
                reasoning_content = msg.get("reasoning_content")
                if reasoning_content:
                    parts.append(types.Part(thought=reasoning_content))
                if content:
                    parts.append(types.Part.from_text(text=content))
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    thought_signature, is_first = msg.get("thought_signature"), True
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name")
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str)
                        except:
                            args = {}
                        tool_call_id_map[tc.get("id")] = name
                        part_kwargs = {
                            "functionCall": types.FunctionCall(
                                name=name,
                                args=args
                            )
                        }
                        if thought_signature and is_first:
                            part_kwargs["thoughtSignature"] = thought_signature
                            is_first = False
                        parts.append(types.Part(**part_kwargs))
                if parts:
                    contents.append(types.Content(
                        role="model",
                        parts=parts
                    ))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                tool_name = tool_call_id_map.get(tool_call_id)
                if tool_name:
                    response_dict = {"result": content}
                    try:
                        if isinstance(content, str):
                             parsed = json.loads(content)
                             if isinstance(parsed, dict):
                                 response_dict = parsed
                        elif isinstance(content, dict):
                             response_dict = content
                    except:
                        pass
                    part = types.Part(function_response=types.FunctionResponse(name=tool_name, response=response_dict))
                    # part = types.Part(function_response=types.FunctionResponse(name=tool_name, response=content))
                    contents.append(types.Content(role="user", parts=[part]))
        return system_instruction, contents

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
        Execute chat using Gemini API.
        Compatible with ChatClient.chat signature.
        """
        system_instruction, contents = self._convert_message_to_gemini(message)
        # 3. Prepare Configuration
        tools_config = kwargs.get("tools") 
        gemini_tools = None
        if tools_config and isinstance(tools_config, list) and isinstance(tools_config[0], dict) and tools_config[0].get("type") == "function":
             gemini_tools = self._convert_openai_tools_to_gemini(tools_config)
        else:
             gemini_tools = tools_config
        config_params = {
            "temperature": 1,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop,
            "system_instruction": system_instruction,
            "tools": gemini_tools
        }
        if "gemini-3" in self.model.lower():
            config_params["thinking_config"] = types.ThinkingConfig(include_thoughts=True,thinking_level=types.ThinkingLevel.HIGH)
        # Handle thinking config
        elif enable_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(include_thoughts=True, thinking_budget=(thinking_budget if (thinking_budget is not None) else -1))
        else:
            config_params['thinking_config'] = types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
        config_params = {k: v for k, v in config_params.items() if v is not None}
        config = types.GenerateContentConfig(**config_params)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            content_text, tool_calls, reasoning_content, thought_signature = "", [], None, None
            finish_reason, usage = "stop", {} # Default
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason:
                    finish_reason = candidate.finish_reason.name # e.g. STOP, MAX_TOKENS
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            content_text += part.text
                        # Extract thought/reasoning if available (Gemini 2.0+ feature)
                        part_thought = getattr(part, "thought", None)
                        # Extract thought_signature if available
                        part_thought_signature = getattr(part, "thought_signature", None)
                        if part_thought_signature:
                            thought_signature = part_thought_signature
                        if part_thought and enable_thinking:
                            if reasoning_content is None:
                                reasoning_content = ""
                            reasoning_content += str(part_thought)
                        if part.function_call:
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}", # Generate a random ID as Gemini doesn't provide one
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(part.function_call.args) if part.function_call.args else "{}"
                                }
                            })
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count
                    }
            # 6. Extract Reasoning (Thinking)
            if enable_thinking and content_text:
                reasoning_content, content_text = self._extract_reasoning_content(content_text)
            return {
                "reasoning_content": reasoning_content,
                "thought_signature": thought_signature,
                "content": content_text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "usage": usage
            }
        except Exception as e:
            print(f"GeminiClient Error: {e}")
            raise e
