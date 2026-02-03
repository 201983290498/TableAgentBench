"""
Global Regex Search Tool
Similar to grep -rn functionality, performs a global search for matching content in files or folders
"""
from typing import List, Dict, Optional
from src.tools.base import BaseTool, ToolResult, register_tool
from src.utils.global_config import GLOBAL_CONFIG
import os
import re
from pathlib import Path
from src.utils.table_process import read_all_sheets_lines


@register_tool(use_cache=True)
class GrepSearchTool(BaseTool):
    """Global regex search tool, searches for matching content in files or folders"""
    
    name = "grep_search"
    ### description = "Type: General Tool. Perform regex search in specified files or folders, returning matching lines and context. Supports csv, xlsx, xls, tsv, txt, py formats. Supports complex Python regex (e.g., grouping, quantifiers, assertions, etc.). By default, no fine-grained retrieval information is passed, first searching for all information in the specified path, and then performing fine-grained retrieval if needed. Fine-grained retrieval is only suitable for file search."
    description = "Type: Precise Table Retrieval. Precise regex search. In specified files or folders, use regex to precisely match strings, returning matching lines and their context information.\n    Returns matching lines. Supported file types include: csv, xlsx, xls, tsv, txt, py formats.\n    Supports complex Python regex (e.g., grouping, quantifiers, assertions, etc.).\n    By default, no fine-grained retrieval information is passed, first searching for all information in the specified path. If too many results are matched to be fully displayed, fine-grained retrieval for a specific file can be performed. Fine-grained retrieval is only suitable for file search.\n   **Applicable Scenarios**: Known exact keyword patterns (e.g., ID format, specific error code), but don't know which file it's in. Compared to keyword search in python_code_executor, it supports powerful regex for pattern matching.\n    **Example**: Find strings like 'ID-1234', pattern can be set to 'ID-\\d{4}'."
    category = "exact_search"
    
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.tsv', '.txt', '.py']
    
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute path to specified file or folder",
            "required": True
        },
        "pattern": {
            "type": "string",
            "description": "Python-style regular expression. Supports complex patterns for precise matching (e.g., 'error|warning', '\\\\d{4}-\\\\d{2}', '(?<=def )\\\\w+', etc.). Please build a regex that can accurately hit the target. This regex will be directly compiled with re.compile to match in files.",
            "required": True
        },
        # "context_lines": {
        #     "type": "integer",
        #     "description": "Number of context lines to preview when relevant content is found. Defaults to 0, representing focusing only on the search content itself.",
        #     "required": False,
        #     "default": 0
        # },
        "file_pattern": {
            "type": "string",
            "description": "Regex filter for filenames to search (only effective when searching folders), e.g., '\\\\.py$', enter a Python-style regex for filtering by filename.",
            "required": False,
            "default": None
        },
        "start_match_idx": {
            "type": "integer",
            "description": "Fine-grained retrieval mode, this parameter only applies to file search. Indicates the index of the matching items to start returning results from, 0 means starting from the first match.",
            "required": False,
            "default": 0
        },
        "max_matches_to_show": {
            "type": "integer",
            "description": "Fine-grained retrieval mode, this parameter only applies to file search. Indicates the maximum number of matching items to return, defaults to 10, None means returning 10. Note: This parameter can only be filled in fine-grained mode!",
            "required": False,
            "default": None
        },
    }
    
    def execute(
        self, 
        path: str, 
        pattern: str, 
        context_lines: int = 0,
        file_pattern: Optional[str] = None,
        start_match_idx: int = 0,
        max_matches_to_show: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute global search"""
        # Initialize fine-grained retrieval parameters (reset on each call)
        self._start_match_idx = 0
        self._max_matches_to_show = None
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return ToolResult(success=False, message=f"Path does not exist: {path}")
        # Fine-grained retrieval mode only applies to single files
        if (start_match_idx > 0 or max_matches_to_show is not None) and os.path.isdir(path):
            return ToolResult(success=False, message="Fine-grained retrieval mode (start_match_idx/max_matches_to_show) only applies to single file retrieval, not supported for folders")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(success=False, message=f"Regular expression error: {e}")
        file_regex = re.compile(file_pattern) if file_pattern else None
        # Collect files to search
        files = self._collect_files(path, file_regex)
        if not files:
            return ToolResult(success=False, message="No searchable files found")
        # Perform search
        results = []
        for file_path in files:
            matches = self._search_file(file_path, regex, context_lines)
            results.extend(matches)
        if not results:
            return ToolResult(True, [], f"No content matching '{pattern}' found")
        # Set fine-grained retrieval parameters (for use in _format_results)
        if start_match_idx > 0 or max_matches_to_show is not None:
            self._start_match_idx = start_match_idx
            self._max_matches_to_show = max_matches_to_show
        return self.make_result(True,results,f"Found {len(results)} matches")
    
    def format_output(self, data) -> str:
        """Format search results"""
        if not data:
            return "No matching results"
        return self._format_results(data)
    
    def _collect_files(self, path: str, file_regex: Optional[re.Pattern] = None) -> List[str]:
        """Collect list of files to search, only supports specified file types"""
        if os.path.isfile(path):
            # Single file: check if extension is supported
            ext = os.path.splitext(path)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                return [path]
            return []
        
        files = []
        for root, _, filenames in os.walk(path):
            # Skip hidden directories and common ignore directories
            if any(skip in root for skip in ['__pycache__', '.git', 'node_modules', '.venv']):
                continue
            for fname in filenames:
                # Check if extension is supported
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue
                if file_regex and not file_regex.search(fname):
                    continue
                files.append(str(Path(os.path.abspath(os.path.join(root, fname)))))
        return files
    
    def _search_file(self, file_path: str, regex: re.Pattern, context_lines: int) -> List[Dict]:
        """Search a single file"""
        try:
            sheets_data = read_all_sheets_lines(file_path)
        except Exception:
            return []
        results = []
        for source_name, lines in sheets_data:
            total = len(lines)
            for idx, line in enumerate(lines):
                if regex.search(line):
                    # Get context
                    context = []
                    for i in range(max(0, idx - context_lines), min(total, idx + context_lines + 1)):
                        context.append({
                            "line_number": i + 1,
                            "content": lines[i].rstrip('\n\r'),
                            "is_match": i == idx
                        })
                    display_path = file_path
                    if "::" in source_name:
                        _, sheet_name = source_name.split("::", 1)
                        display_path = f"{file_path}::{sheet_name}"
                    results.append({
                        "file": display_path,
                        "line_number": idx + 1,
                        "match_content": line.rstrip('\n\r'),
                        "context": context
                    })
        return results
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format output - Intelligent Compressed Version"""
        if not results:
            return "No matching results"
        # Read constants from global configuration
        max_detail_files = GLOBAL_CONFIG.get("tools", {}).get("grep_search", {}).get("max_detail_files", 10)
        max_matches_per_file = GLOBAL_CONFIG.get("tools", {}).get("grep_search", {}).get("max_matches_per_file", 5)
        max_fine_grained_matches = GLOBAL_CONFIG.get("tools", {}).get("grep_search", {}).get("default_fine_grained_matches_to_show", 10)
        # Get fine-grained retrieval parameters
        start_match_idx = getattr(self, '_start_match_idx', 0)
        max_matches_to_show = getattr(self, '_max_matches_to_show', None)
        is_fine_grained = start_match_idx > 0 or max_matches_to_show is not None
        # Record original total
        total_matches_original = len(results)
        # Fine-grained mode: slice the results
        if is_fine_grained:
            # Limit maximum return count for fine-grained retrieval
            max_matches_to_show = max_matches_to_show or max_fine_grained_matches
            end_idx = start_match_idx + max_matches_to_show
            results = results[start_match_idx:end_idx]
            # No more truncation in fine-grained mode
            max_matches_per_file = len(results)
        # Group by file
        file_groups = {}
        for item in results:
            file_path = item['file']
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(item)
        total_files = len(file_groups)
        total_matches = len(results)
        output = []
        
        # Summary information
        output.append("=" * 40)
        output.append("Search Result Overview")
        output.append("=" * 40)
        output.append(f"• Matching files: {total_files}")
        if is_fine_grained:
            shown_start = start_match_idx + 1
            shown_end = start_match_idx + total_matches
            output.append(f"• Total matches: {total_matches_original}, currently showing {shown_start}-{shown_end}")
        else:
            output.append(f"• Total matches: {total_matches}")
        output.append("=" * 40)
        output.append("")
        
        # Sort files by match count
        sorted_files = sorted(file_groups.items(), key=lambda x: len(x[1]), reverse=True)
        # Detailed display of top N files
        for idx, (file_path, matches) in enumerate(sorted_files[:max_detail_files]):
            match_count = len(matches)
            show_count = min(match_count, max_matches_per_file)
            if match_count <= max_matches_per_file:
                status = f"(All shown)"
            else:
                status = f"(Showing first {show_count})"
            output.append(f"【File {idx + 1}/{total_files}】{file_path} - {match_count} matches {status}")
            output.append("─" * 40)
            # Merge context lines of adjacent matches to avoid duplicate display
            merged_lines = self._merge_context_lines(matches[:max_matches_per_file])
            for line_info in merged_lines:
                if line_info['line_number'] == 0:  # Separator line
                    output.append("    ......")
                else:
                    marker = ">>>" if line_info['is_match'] else "   "
                    output.append(f"{marker} {line_info['line_number']:5d} │ {line_info['content']}")
            output.append("")
        # Remaining files overview
        remaining_files = sorted_files[max_detail_files:]
        if remaining_files:
            output.append("─" * 40)
            output.append(f"Matching overview for remaining {len(remaining_files)} files:")
            output.append("─" * 40)
            for file_path, matches in remaining_files:
                output.append(f"• {file_path} - {len(matches)} matches")
        
        return "\n".join(output)
    
    def _merge_context_lines(self, matches: List[Dict]) -> List[Dict]:
        """Merge context lines of adjacent matches to avoid duplicate display"""
        if not matches:
            return []
        # Collect all line information, deduplicate by line number
        lines_dict = {}
        for match in matches:
            for ctx in match['context']:
                line_num = ctx['line_number']
                # If this line already exists, keep the is_match=True status
                if line_num in lines_dict:
                    if ctx['is_match']:
                        lines_dict[line_num]['is_match'] = True
                else:
                    lines_dict[line_num] = {
                        'line_number': line_num,
                        'content': ctx['content'],
                        'is_match': ctx['is_match']
                    }
        
        # Sort by line number
        sorted_lines = sorted(lines_dict.values(), key=lambda x: x['line_number'])
        # Insert empty line markers between non-continuous lines
        result = []
        prev_line_num = None
        for line_info in sorted_lines:
            if prev_line_num is not None and line_info['line_number'] > prev_line_num + 1:
                # Non-continuous, insert empty line separator
                result.append({'line_number': -1, 'content': '', 'is_match': False})
            result.append(line_info)
            prev_line_num = line_info['line_number']
        
        # Filter out empty line markers, replace with actual empty lines
        final_result = []
        for item in result:
            if item['line_number'] == -1:
                # Add an empty line as a separator
                if final_result:  # Avoid adding empty line at the beginning
                    final_result.append({'line_number': 0, 'content': '...', 'is_match': False})
            else:
                final_result.append(item)
        return final_result
