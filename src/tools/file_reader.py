"""
Table File Reading Tool
Read the first N lines of a table file
"""
from typing import Optional
from pathlib import Path
import os

from src.tools.base import BaseTool, ToolResult, register_tool
from src.utils.table_process import read_all_sheets_lines

@register_tool(use_cache=False)
class TableHeadReader(BaseTool):
    """
    Table Head Reader Tool
    Reads the first N lines of a specified table file for quick preview of table data
    """
    
    name = "table_head_reader"
    description = "Type: Table Processing. Reads specified lines from a table file. Requires start line and row count, returns file path, total rows, and content (line number + content, line numbers start from 1). Used for quick preview of table data structure."
    category = "table_process"
    
    parameters = {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the table file (supports csv, xlsx, xls).",
            "required": True
        },
        "start": {
            "type": "integer",
            "description": "Starting line number (1-indexed), defaults to 1",
            "required": False
        },
        "n": {
            "type": "integer",
            "description": "Number of lines to read starting from the start line, defaults to 10",
            "required": False
        }
    }
    
    def execute(
        self, 
        file_path: str,
        start: int = 1,
        n: int = 10,
        **kwargs
    ) -> ToolResult:
        """
        Execute table head reading
        
        Args:
            file_path: Path to the table file
            n: Number of lines to read, defaults to 5
            
        Returns:
            ToolResult: Contains file path, total rows, and first N lines of content
        """
        # Validate file path
        path = Path(os.path.abspath(file_path))
        file_path = str(path)
        if not path.exists():
            return ToolResult(success=False,  message=f"File does not exist: {file_path}")
        # Check if it's a directory
        if path.is_dir():
            return ToolResult(success=False, message=f"Path is a directory: {file_path}. Please use table_locator tool to view folder contents, or provide a specific file path.")

        all_sheets = read_all_sheets_lines(file_path)
        if not all_sheets:
            return ToolResult(success=False, message=f"Unable to read file content or file is empty: {file_path}")
        # Get data from the first sheet
        _, lines = all_sheets[0]
        # Check for other sheets
        other_sheets = []
        if len(all_sheets) > 1:
            for name, _ in all_sheets[1:]:
                if "::" in name:
                    other_sheets.append(name.split("::", 1)[1])
                else:
                    other_sheets.append(name)
        warning_msg = ""
        if other_sheets:
            warning_msg = f"\n[Note] Detected multiple sub-sheets in this file: {', '.join(other_sheets)}. Currently only showing the first sub-sheet. It is recommended to split the multi-sheet file to get complete information."
        total_rows = len(lines)
        # Correct start line number to prevent 0 or negative input
        start = max(1, start)
        head_lines = lines[start-1:start-1+n]
        # Original data
        ori_data = {
            "file_path": file_path,
            "file_name": path.name,
            "total_rows": total_rows,
            "start": start,
            "lines": head_lines,
            "warning": warning_msg
        }
        return self.make_result(
            success=True,
            data=ori_data,
            message=f"Preview {path.name}: Lines {start}-{start+len(head_lines)-1} / Total {total_rows} lines" + warning_msg
        )
    
    def format_output(self, data) -> str:
        """Format file preview"""
        if not data:
            return ""
        path_name = data.get("file_name", "")
        total = data.get("total_rows", 0)
        start = data.get("start", 1)
        lines = data.get("lines", [])
        n = len(lines)
        
        content_lines = []
        for i, line in enumerate(lines, start=start):
            content_lines.append(f"{i:3d}| {line.rstrip()}")
        
        remaining = f" (+{total - start - n + 1} more)" if total > start + n - 1 else ""
        warning = data.get("warning", "")
        return (
            f"[{path_name}] Total {total} lines{remaining}\n"
            f"{'─' * 40}\n"
            + "\n".join(content_lines)
            + warning
        )
