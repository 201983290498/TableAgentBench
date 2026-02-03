"""
Tool base class and registration mechanism
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
import os
import hashlib
from functools import wraps

# Use path relative to the current file to ensure it can run in different environments
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp")
CACHE_FILE = os.path.join(CACHE_DIR, "tool_cache.json")
TOOL_CACHE: Dict[str, Any] = {}
CACHE_LOADED = False

def load_cache():
    global TOOL_CACHE, CACHE_LOADED
    if CACHE_LOADED:
        return
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                TOOL_CACHE = json.load(f)
        except Exception:
            pass
    CACHE_LOADED = True

def save_cache():
    global TOOL_CACHE
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(TOOL_CACHE, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"Error saving cache: {e}")


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    data: Any = None           # Formatted output (for LLM)
    ori_data: Any = None       # Original data (for program use)
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "ori_data": self.ori_data,
            "message": self.message
        }
    
    def __str__(self) -> str:
        if self.success:
            return f"[Success] {self.message}\n{self.data}" if self.data else f"[Success] {self.message}"
        return f"[Failed] {self.message}\n{self.data}" if self.data else f"[Failed] {self.message}"


class BaseTool(ABC):
    """
    Tool base class
    All tools must inherit from this class and implement the execute method
    """
    name: str = "base_tool"
    description: str = "Type: General Tool. Tool description"
    category: str = "general"  # Tool type: location, understanding, retrieval, reasoning, general
    parameters: Dict[str, Any] = {}
    
    def get_llm_client(self, llm_client: Any = None) -> Any:
        """
        Get LLM client - simplified version
        
        Args:
            llm_client: Optional, directly passed client
            
        Returns:
            LLM client instance
        """
        if llm_client is not None:
            return llm_client
        
        # Use global singleton
        from src.utils.chat_api import get_chat_client
        return get_chat_client()
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult: Execution result
        """
        pass
    
    def format_output(self, data: Any) -> str:
        """
        Format output for LLM reading
        Subclasses can override this method to implement custom formatting
        
        Args:
            data: Original data
            
        Returns:
            Formatted string
        """
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, (list, dict)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)
    
    def make_result(self, success: bool, data: Any = None, message: str = "") -> ToolResult:
        """
        Create tool result, automatically call format_output - simplified version
        
        Args:
            success: Whether successful
            data: Original data
            message: Message
            
        Returns:
            ToolResult
        """
        return ToolResult(
            success=success, 
            data=self.format_output(data), 
            ori_data=data, 
            message=message
        )
    
    def validate_params(self, **kwargs) -> bool:
        """Validate if parameters are valid"""
        required_params = [
            k for k, v in self.parameters.items() 
            if isinstance(v, dict) and v.get("required", False)
        ]
        for param in required_params:
            if param not in kwargs:
                return False
        return True
    
    def to_function_schema(self) -> Dict:
        """
        Convert to OpenAI function calling format
        For LLM tool calling
        """
        properties = {}
        required = []
        
        for param_name, param_info in self.parameters.items():
            if isinstance(param_info, dict):
                properties[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", "")
                }
                if param_info.get("enum"):
                    properties[param_name]["enum"] = param_info["enum"]
                if param_info.get("items"):
                    properties[param_name]["items"] = param_info["items"]
                if param_info.get("required", False):
                    required.append(param_name)
            else:
                properties[param_name] = {"type": "string", "description": str(param_info)}
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


# Tool registry
TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tool(use_cache: bool = False):
    """
    Tool registration decorator
    
    Args:
        use_cache: Whether to enable result caching, defaults to False
    
    Usage:
    @register_tool(use_cache=True)  # Enable caching
    class MyTool(BaseTool):
        name = "my_tool"
        ...
        
    @register_tool()  # Do not enable caching (can also be written as @register_tool)
    class MyTool(BaseTool):
        name = "my_tool"
        ...
    """
    def decorator(tool_class: type) -> type:
        instance = tool_class()
        
        if use_cache:
            # ------------------ Cache Logic Start ------------------
            original_execute = instance.execute

            @wraps(original_execute)
            def cached_execute(**kwargs):
                load_cache()
                # Generate cache key
                try:
                    # Sort keys to ensure consistency
                    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
                except Exception:
                    kwargs_str = str(kwargs) # Fallback if json fails
                
                h = hashlib.md5(kwargs_str.encode()).hexdigest()
                key = f"{instance.name}_{h}"
                
                # Check cache
                if key in TOOL_CACHE:
                    try:
                        return ToolResult(**TOOL_CACHE[key])
                    except Exception:
                        pass
                
                # Execute original method
                result = original_execute(**kwargs)
                
                # Save cache
                TOOL_CACHE[key] = result.to_dict()
                save_cache()
                
                return result

            # Replace the instance method
            instance.execute = cached_execute
            # ------------------ Cache Logic End ------------------
        
        TOOL_REGISTRY[instance.name] = instance
        return tool_class
    
    return decorator


def get_tool(name: str) -> Optional[BaseTool]:
    """Get tool by name"""
    return TOOL_REGISTRY.get(name)


def get_all_tools(include_tools: Optional[List[str]] = None) -> List[BaseTool]:
    """Get all registered tools"""
    if include_tools is None:
        return list(TOOL_REGISTRY.values())
    return [t for name, t in TOOL_REGISTRY.items() if name in include_tools]


def get_tools_by_category(category: str) -> List[BaseTool]:
    """Get tools by category"""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


def get_tools_grouped_by_category(include_tools: Optional[List[str]] = None) -> Dict[str, List[BaseTool]]:
    """Get tools grouped by category"""
    from src.tools.tool_category import ToolCategory
    result = {cat.value: [] for cat in ToolCategory}
    for name, tool in TOOL_REGISTRY.items():
        if include_tools is not None and name not in include_tools:
            continue
        cat = tool.category if tool.category in result else "general"
        result[cat].append(tool)
    return result


def get_tools_schema(include_tools: Optional[List[str]] = None) -> List[Dict]:
    """Get function calling schema for all tools"""
    if include_tools is None:
        return [tool.to_function_schema() for tool in TOOL_REGISTRY.values()]
    return [tool.to_function_schema() for name, tool in TOOL_REGISTRY.items() if name in include_tools]

def get_tools_schema_by_category(category: str) -> List[Dict]:
    """Get tool schema by category"""
    return [tool.to_function_schema() for tool in TOOL_REGISTRY.values() if tool.category == category]


def execute_tool(tool_name: str, **kwargs) -> ToolResult:
    """Execute specified tool - simplified version"""
    tool = get_tool(tool_name)
    if tool is None:
        return ToolResult(success=False, message=f"Tool {tool_name} does not exist")
    
    if not tool.validate_params(**kwargs):
        return ToolResult(success=False, message=f"Tool {tool_name} parameter validation failed")
    
    try:
        return tool.execute(**kwargs)
    except Exception as e:
        return ToolResult(success=False, message=f"Tool execution exception: {str(e)}")
