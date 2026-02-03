import json
import csv
import ast
from typing import Any, Dict, List, Optional
import re
import os
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

def format_conversation_history(
    conversation_history: List[Dict[str, str]],
    max_turns: int = 6,
    max_content_length: int = 200,
    user_label: str = "User",
    assistant_label: str = "Assistant",
    include_header: bool = True,
    format_style: str = "text"
) -> str:
    """
    Format conversation history into prompt string
    
    Args:
        conversation_history: List of conversation history, format [{"role": "user/assistant", "content": "..."}]
        max_turns: Maximum number of conversation turns to include (recent N)
        max_content_length: Maximum character length per message (truncate if exceeded)
        user_label: Display label for user role
        assistant_label: Display label for assistant role
        include_header: Whether to include header line
        format_style: Format style, supports "text" (plain text), "markdown" (markdown format), "xml" (xml tag format)
        
    Returns:
        Formatted conversation history string
    
    Example:
        >>> history = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hello, how can I help you?"}
        ... ]
        >>> print(format_conversation_history(history))
        Conversation History:
        User: Hello
        Assistant: Hello, how can I help you?
    """
    if not conversation_history:
        return ""
    # Take recent N turns
    recent_history = conversation_history[-max_turns:] if max_turns else conversation_history
    # Role mapping
    role_map = {
        "user": user_label,
        "assistant": assistant_label,
        "system": "System",
        "tool": "Tool"
    }
    lines = []
    # Add title based on format style
    if include_header:
        if format_style == "markdown":
            lines.append("### Conversation History\n")
        elif format_style == "xml":
            lines.append("<conversation_history>")
        else:
            lines.append("Conversation History:")
    
    # Format each message
    for msg in recent_history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # Handle tool_calls and tool result
        if role == 'assistant' and 'tool_calls' in msg:
            tool_calls = msg.get('tool_calls', [])
            tool_info = []
            for tc in tool_calls:
                func_name = tc.get('function', {}).get('name', 'unknown')
                args = tc.get('function', {}).get('arguments', '{}')
                tool_info.append(f"Call tool: {func_name}({args})")
            if content:
                content += "\n" + "\n".join(tool_info)
            else:
                content = "\n".join(tool_info)
        
        # Truncate content that is too long
        if max_content_length and len(content) > max_content_length:
            content = content[:max_content_length]
        label = role_map.get(role, role)
        if format_style == "markdown":
            lines.append(f"**{label}**: {content}\n")
        elif format_style == "xml":
            lines.append(f"  <message role=\"{role}\">{content}</message>")
        else:
            lines.append(f"{label}: {content}")
    # Close xml tag
    if format_style == "xml" and include_header:
        lines.append("</conversation_history>")
    return "\n".join(lines)



def read_json_file(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
    return data

def read_text_file(path: str) -> List[str]:
    """
    Read text file, automatically try multiple encodings.
    """
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin-1']
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.readlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
            
    # Finally try reading in binary mode and ignore errors.
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.readlines()
                            
def parse_json_response(response: str) -> Any:
    """
    Parse JSON content from string, supports object {} and array [] formats.
    Automatically handles markdown code block wrapping.
    """
    if not response:
        return {"error": "Empty response"}
    
    text = response.strip()
    
    # 1. First attempt to extract content within ```json ... ``` code blocks.
    # Strategy A: Non-greedy match (try first, handle standard cases and multiple code blocks).
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_match = re.search(code_block_pattern, text)
    if code_match:
        content = code_match.group(1).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
            
    # Strategy B: Greedy match (handle nested ``` cases, e.g., mermaid graph).
    code_block_pattern_greedy = r'```(?:json)?\s*([\s\S]*)\s*```'
    code_match_greedy = re.search(code_block_pattern_greedy, text)
    
    if code_match_greedy:
        content_greedy = code_match_greedy.group(1).strip()
        try:
            return json.loads(content_greedy)
        except json.JSONDecodeError:
            pass

    candidates = []
    if code_match:
        candidates.append(code_match.group(1).strip())
    if code_match_greedy:
        greedy_content = code_match_greedy.group(1).strip()
        if not candidates or greedy_content != candidates[0]:
            candidates.append(greedy_content)
    
    # If no code block is found, use the original text.
    if not candidates:
        candidates.append(text)
        
    for content in candidates:
        # 2. Try direct parsing (actually already tried above, but if candidates include raw text, it's tried here).
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 3. Attempt to extract JSON object {...}
        obj_pattern = r'\{[\s\S]*\}'
        obj_match = re.search(obj_pattern, content)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # 4. Attempt to extract JSON array [...]
        arr_pattern = r'\[[\s\S]*\]'
        arr_match = re.search(arr_pattern, content)
        if arr_match:
            try:
                return json.loads(arr_match.group(0))
            except json.JSONDecodeError:
                pass
    
    return {"error": "Invalid JSON format", "raw": response}

def read_config(config_path: str = None) -> Dict[str, Any]:
    cfg_path = config_path or os.environ.get("TABLE_AGENT_CONFIG_PATH")
    if not cfg_path:
        cfg_path = str(Path(__file__).resolve().parent.parent.parent / "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_model_cache_dir(config_path: str = None) -> Optional[str]:
    cfg = read_config(config_path)
    return cfg.get("model_cache_dir")

def get_default_embedding_model(config_path: str = None) -> Optional[str]:
    cfg = read_config(config_path)
    return cfg.get("default_embedding_model")


def get_default_device():
    """
    Automatically detect and return available device type
    Priority: CUDA > ROCm > NPU > MPS > CPU
    """
    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "hip") and torch.version.hip is not None:
        return "cuda"  # ROCm also uses "cuda" device type in PyTorch
    elif hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_dynamic_batch_size(table_docs: List[str]) -> int:
    """
    Dynamically calculate batch_size based on the number of tokens in documents.
    """
    from src.utils.chat_api import ChatClient  # Local import to avoid circular reference

    # Dynamically set batch_size
    max_tokens = 0
    if table_docs:
        token_counts = ChatClient.count_tokens(table_docs)
        if isinstance(token_counts, int):
            token_counts = [token_counts]
        if token_counts:
            max_tokens = max(token_counts)
    
    batch_size = 64
    if max_tokens > 4096:
        batch_size = 4
    elif max_tokens > 2048:
        batch_size = 8
    elif max_tokens > 1024:
        batch_size = 16
    elif max_tokens > 512:
        batch_size = 32
        
    return batch_size
