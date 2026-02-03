"""
Tool System
"""
from src.tools.base import (
    BaseTool,
    ToolResult,
    register_tool,
    get_tool,
    get_all_tools,
    get_tools_by_category,
    get_tools_grouped_by_category,
    get_tools_schema,
    execute_tool,
    TOOL_REGISTRY
)

from src.tools.tool_category import (
    ToolCategory,
    CATEGORY_INFO,
    format_tools_description,
    get_category_description
)

# Import tools to trigger registration
from src.tools.table_locator import TableLocator, TableSelector
from src.tools.complex_table_parser_v2 import ComplexTableParserV2
from src.tools.header_merger import HeaderMerger
from src.tools.xlsx_to_csv_converter import XlsxToCsvConverter
from src.tools.file_reader import TableHeadReader
from src.tools.row_retriever import SemanticRowRetriever
from src.tools.column_retriever import SemanticColumnRetriever
from src.tools.code_generator import PythonCodeExecutor
from src.tools.cmd_executor import CmdExecutor
from src.tools.grep_search_tool import GrepSearchTool

__all__ = [
    # Base Components
    "BaseTool",
    "ToolResult",
    "register_tool",
    "get_tool",
    "get_all_tools",
    "get_tools_by_category",
    "get_tools_grouped_by_category",
    "get_tools_schema",
    "execute_tool",
    "TOOL_REGISTRY",
    # Category Related
    "ToolCategory",
    "CATEGORY_INFO",
    "format_tools_description",
    "get_category_description",
    # Location Tools
    "TableLocator",
    "TableSelector",
    # Table Tools
    "ComplexTableParserV2",
    "HeaderMerger",
    "XlsxToCsvConverter",
    "TableHeadReader",
    # Retrieval Tools
    "SemanticRowRetriever",
    "SemanticColumnRetriever",
    # Reasoning Tools
    "PythonCodeExecutor",
    # General Tools
    "CmdExecutor",
    "GrepSearchTool",
]
