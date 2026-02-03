"""
Tool Category Management
Defines the functional types of tools, for the model to choose freely as needed
"""
from enum import Enum
from typing import List, Dict, Any, Optional


class ToolCategory(Enum):
    """Tool functional categories"""
    LOCATION = "location"           # File location category
    UNDERSTANDING = "understanding" # Table understanding/preprocessing category
    RETRIEVAL = "retrieval"         # Content retrieval/recall category
    REASONING = "reasoning"         # Programming/calculation category
    GENERAL = "general"             # General utility category


# Tool category information
CATEGORY_INFO: Dict[ToolCategory, Dict[str, str]] = {
    ToolCategory.LOCATION: {
        "name": "File Location",
        "desc": "These tools are mainly used to locate table files from folder structure previews or semantic search perspectives.",
    },
    ToolCategory.UNDERSTANDING: {
        "name": "Table Understanding",
        "desc": "These tools are mainly used for table preview, preprocessing complex tables into simple tables suitable for pandas analysis, and understanding table structure and content.",
    },
    ToolCategory.RETRIEVAL: {
        "name": "Content Retrieval",
        "desc": "These tools are mainly based on semantic retrieval, helping to locate relevant content within tables. They help in understanding user intent and finding relevant data.",
    },
    ToolCategory.REASONING: {
        "name": "Table Reasoning",
        "desc": "These tools are mainly used for pandas programming, data calculation, and processing.",
    },
    ToolCategory.GENERAL: {
        "name": "General Tools",
        "desc": "General auxiliary functions, tools that can be used at any time.",
    },
}


def format_tools_description(tools_by_category: Optional[Dict[str, List[Any]]] = None) -> str:
    """Format tool descriptions (displayed grouped by type - detailed version)"""
    if tools_by_category is None:
        from src.tools.base import get_tools_grouped_by_category
        tools_by_category = get_tools_grouped_by_category()
    
    lines = ["Here are the available tools, grouped by category (can be used freely as needed):", ""]
    
    for idx, category in enumerate(ToolCategory, 1):
        tools = tools_by_category.get(category.value, [])
        if not tools:
            continue
        
        info = CATEGORY_INFO.get(category, {})
        lines.append(f"## {idx}. {info.get('name', category.value)} - {info.get('desc', '')}")
        lines.append("")
        
        for tool in tools:
            # Use H3 to highlight tool name
            lines.append(f"### `{tool.name}`")
            lines.append(f"**Description**: {tool.description}")
            
            if tool.parameters:
                lines.append("**Parameters**:")
                for k, v in tool.parameters.items():
                    # Get detailed information
                    # Compatibility handling: ensure v is a dictionary
                    if isinstance(v, dict):
                        p_type = v.get('type', 'string')
                        p_desc = v.get('description', '')
                        # base.py defaults required to False, keeping consistent here
                        p_req = "required" if v.get('required', False) else "optional"
                    else:
                        p_type = "string"
                        p_desc = str(v)
                        p_req = "optional"
                    
                    # Assemble parameter description line
                    lines.append(f"- `{k}` ({p_type}, {p_req}): {p_desc}")
            else:
                lines.append("**Parameters**: None")
            lines.append("")
    
    return "\n".join(lines) if len(lines) > 2 else "No tools currently available"


def get_category_description() -> str:
    """Get detailed description of tool categories"""
    lines = ["Tool Category Description:"]
    for cat in ToolCategory:
        name = CATEGORY_INFO.get(cat, {}).get("name", cat.value)
        desc = CATEGORY_INFO.get(cat, {}).get("desc", "")
        lines.append(f"- **{name}** ({cat.value}): {desc}")
    return "\n".join(lines)