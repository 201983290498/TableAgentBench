"""
Semantic Row Retrieval Tool
Retrieves rows from table files most relevant to the query based on vector similarity
Supported formats: csv, xlsx, xls, tsv
"""
from typing import List, Optional
from pathlib import Path
from src.tools.base import BaseTool, ToolResult, register_tool
from src.retrival.embedder import semantic_search
from src.utils.table_process import scan_table_files, read_table_lines
from src.utils.global_config import dataset_list
from src.utils.chat_api import ChatClient
from src.utils.common import get_dynamic_batch_size
import os

@register_tool()
class SemanticRowRetriever(BaseTool):
    """
    Semantic Row Retrieval Tool
    Retrieves most relevant rows based on embedding similarity in table files
    Supports: csv, xlsx, xls, tsv
    """
    
    name = "semantic_row_retriever"
    ### description = "Type: Content Retrieval. Performs semantic search in specified table files or folders, retrieving most relevant rows. Supports csv, tsv formats. Please enter a complete query."
    description = """Type: Semantic Retrieval. Performs semantic search for query-related data rows in specified files or folders. Matches semantic patterns between specific row data and questions. Unlike table locator, this enables finer-grained row data positioning.
**Applicable Scenarios**: 1. When tables have extremely many rows (long tables) that cannot be read at once, and rows containing specific semantic information (not exact keywords) need to be found. 2. When the question cannot be accurately retrieved using keyword search.
**Example**: Input query='Find sales records related to iPhone 15 Pro', can match rows containing relevant descriptions."""
    category = "semantic_search"
    cache_file = "row_retriever_cache.pkl"
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.tsv']
    
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute path to table files or folders (supports csv, tsv)",
            "required": True
        },
        "query": {
            "type": "string",
            ### "description": "Query text",
            "description": "Query text. It's recommended to use complete natural language describing characteristics of the query content, e.g., 'Find sales records containing Huawei Mate 60'. Richer query information helps improve retrieval accuracy.",
            "required": True
        },
        "top_k": {
            "type": "integer",
            "description": "Number of rows to return, defaults to 10",
            "required": False
        }
    }
    
    def __init__(self):
        pass
    
    def _collect_table_files(self, path: str) -> List[Path]:
        """Collect table files: return directly if it's a file, scan if it's a folder"""
        p = Path(path)
        if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            return [p]
        elif p.is_dir():
            infos = scan_table_files(path, extensions=self.SUPPORTED_EXTENSIONS)
            return [Path(i['path']) for i in infos]
        else:
            return []
    
    def _load_and_prepare(self, table_path: Path) -> List[dict]:
        """Load tables and prepare row data, supports csv, xlsx, xls, tsv"""
        lines, _, total = read_table_lines(str(table_path), mode="all")
        if not lines:
            return []
        
        # First row is the header
        header = lines[0].split(',')
        rows_data = []
        
        for i, line in enumerate(lines[1:], start=2):
            values = line.split(',')
            # Join header and values: "col1: val1 | col2: val2 | ..."
            text_parts = [f"{header[i]}: {values[i]}" for i in range(min(len(header), len(values)))]
            rows_data.append({
                "table": str(table_path),
                "header": header,
                "row": values,
                "line_number": i,
                "text": f"Table path: {str(table_path)}\n" + " | ".join(text_parts)
            })
        
        return rows_data
    
    def execute(self, path: str, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        """Execute semantic row retrieval"""
        path = os.path.abspath(path)
        if len([item for item in dataset_list if item in path])>0: 
            cache_path = os.path.join([item for item in dataset_list if item in path][0], self.cache_file)
        else:
            cache_path = None
        # 1. Collect table files
        table_files = self._collect_table_files(path)
        if not table_files:
            return ToolResult(success=False, message=f"No supported table files found: {path}")
        
        # Distinguish between processed and ignored files
        processing_files = []
        skipped_files = []
        for f in table_files:
            if f.suffix.lower() in ['.csv', '.tsv']:
                processing_files.append(f)
            elif f.suffix.lower() in ['.xlsx', '.xls']:
                skipped_files.append(f)
        
        # 2. Load all row data
        all_rows = []
        for table_file in processing_files:
            try:
                rows = self._load_and_prepare(table_file)
                all_rows.extend(rows)
            except Exception as e:
                continue  # Skip files that fail to read
        
        if not all_rows and not skipped_files:
            return ToolResult(success=False, message="Unable to read any table data")
        
        # 3. Semantic retrieval
        results = []
        if all_rows:
            texts = ["Row content in table, joined in col: value format: " + r["text"] for r in all_rows]
            if len(texts) >= 100000:
                return ToolResult(success=False, message="More than 100,000 rows in the retrieval path, cannot perform retrieval. Please narrow down the scope or use other tools.")
            batch_size = get_dynamic_batch_size(texts)
            results_sim = semantic_search(f"Task: You need to retrieve the corresponding rows in the table based on the query: {query}", texts, top_k, cache_path=cache_path, batch_size=batch_size)
            
            # 5. Build results
            for r in results_sim:
                item = all_rows[r.index]
                results.append({
                    "table": item["table"],
                    "header": item["header"],
                    "row": item["row"],
                    "line_number": item.get("line_number", "?"),
                    "score": float(r.score)
                })
        
        return self.make_result(
            success=True,
            data={"results": results, "skipped_files": [str(f) for f in skipped_files]},
            message=f"Retrieved {len(results)} results from {len(all_rows)} rows in {len(processing_files)} files"
        )
    
    def format_output(self, data) -> str:
        """Format retrieval results"""
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            skipped_files = data.get("skipped_files", [])
        else:
            results = data
            skipped_files = []

        output_lines = []

        if not results:
            output_lines.append("No retrieval results")
        else:
            for item in results:
                output_lines.append(f"Data path: {item['table']} Similarity: {item['score']}")
                output_lines.append(f"Header: {','.join(str(h) for h in item['header'])}")
                output_lines.append(f"Line number {item.get('line_number', '?')}: {','.join(str(v) for v in item['row'])}")
                output_lines.append("")

        # Add warning for skipped files
        if skipped_files:
            output_lines.append(f"\nWarning: There are {len(skipped_files)} xlsx or xls table files. Currently, only retrieval for csv and tsv formats is supported.")
            if len(skipped_files) > 3:
                show_files = skipped_files[:3]
                output_lines.append("Paths for first 3 files:")
            else:
                show_files = skipped_files
                output_lines.append("File paths:")
            
            for f in show_files:
                rel_path = f
                output_lines.append(f"- {rel_path}")
        
        return "\n".join(output_lines)


# @register_tool()
class SemanticRowRetriever2(BaseTool):
    """
    Semantic Row Retrieval Tool
    Retrieves most relevant rows based on embedding similarity in table files
    Supports: csv, xlsx, xls, tsv
    """
    
    name = "semantic_row_retriever"
    ### description = "Type: Content Retrieval. Performs semantic search in specified table files or folders, retrieving most relevant rows. Supports csv, tsv formats. Please enter a complete query."
    description = "Type: Content Retrieval. Semantic retrieval for table rows in specified files or folders, intelligently retrieves row-level content based on deep semantic understanding, breaking through traditional keyword search limitations to locate semantically relevant rows.\n    *   **Applicable Scenarios** 1. Extra-large table processing: efficient and accurate retrieval for massive tables (100k+ records) that cannot be fully loaded or traversed.\n 2. Multilingual retrieval: supports cross-language semantic matching, retrieving relevant records without needing exact translation.\n 3. Fuzzy semantic query: handles cases where user query and table content differ in expression but share semantic meaning (e.g., 'high-end smartphone sales' matches 'iPhone 15 Pro sales data').\n 4. Terminology conversion: automatically understands relationships between industry terms, synonyms, abbreviations, and full names (e.g., 'heart attack' matches 'myocardial infarction' records).\n   **Example**: Input query='Find sales records related to iPhone 15 Pro', can match rows containing relevant descriptions."
    category = "retrieval"
    cache_file = "row_retriever_cache.pkl"
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.tsv']
    
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute path to table files or folders (supports csv, tsv)",
            "required": True
        },
        "query": {
            "type": "string",
            ### "description": "Query text",
            "description": "Query text. It's recommended to use complete natural language, e.g., 'Find sales records containing Huawei Mate 60'.",
            "required": True
        },
        "top_k": {
            "type": "integer",
            "description": "Number of rows to return, defaults to 5",
            "required": False
        }
    }
    
    def __init__(self):
        pass
    
    def _collect_table_files(self, path: str) -> List[Path]:
        """Collect table files: return directly if it's a file, scan if it's a folder"""
        p = Path(path)
        if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            return [p]
        elif p.is_dir():
            infos = scan_table_files(path, extensions=self.SUPPORTED_EXTENSIONS)
            return [Path(i['path']) for i in infos]
        else:
            return []
    
    def _load_and_prepare(self, table_path: Path) -> List[dict]:
        """Load table and prepare row data, supports csv, xlsx, xls, tsv"""
        lines, _, total = read_table_lines(str(table_path), mode="all")
        if not lines:
            return []
        rows_data = []
        # First row is header
        header_info = '\n'.join(lines[:2])
        header = f"Header data:\n{header_info}\n-------\nSpecific table data:\n" 
        header_tokens = ChatClient.count_tokens(header)
        tmp_block = []
        now_tokens = header_tokens
        for i, line in enumerate(lines):
            if i < 2: continue # Skip first 3 header rows (wait, code says i < 2)
            line_token = ChatClient.count_tokens(line)
            # If current line plus existing block exceeds limit, and block is not empty -> save block
            if now_tokens + line_token > 512 and tmp_block:
                rows_data.append({
                    "table": str(table_path),
                    "header": header,
                    "row": i - len(tmp_block) + 1,
                    "text": header + "\n".join(tmp_block)
                })
                # Reset block, keep last line as overlap
                last_line = tmp_block[-1]
                tmp_block = [last_line]
                now_tokens = header_tokens + ChatClient.count_tokens(last_line)
            # Add current line
            tmp_block.append(line)
            now_tokens += line_token
        # Save remaining block
        if tmp_block:
            rows_data.append({
                "table": str(table_path),
                "header": header,
                "row": len(lines) - len(tmp_block) + 1,
                "text": header + "\n".join(tmp_block)
            })
            
        return rows_data
    
    def execute(self, path: str, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """Execute semantic row retrieval"""
        path = os.path.abspath(path)
        if len([item for item in dataset_list if item in path])>0: 
            cache_path = os.path.join([item for item in dataset_list if item in path][0], self.cache_file)
        else:
            cache_path = None
        # 1. Collect table files
        table_files = self._collect_table_files(path)
        if not table_files:
            return ToolResult(success=False, message=f"No supported table files found: {path}")
        # 2. Distinguish between processed and ignored files
        processing_files, skipped_files = [], []
        for f in table_files:
            if f.suffix.lower() in ['.csv', '.tsv']:
                processing_files.append(f)
            elif f.suffix.lower() in ['.xlsx', '.xls']:
                skipped_files.append(f)
        # 3. Load all row data
        all_rows = []
        for table_file in processing_files:
            try:
                rows = self._load_and_prepare(table_file)
                all_rows.extend(rows)
            except Exception as e:
                continue  # Skip files that fail to read
        if not all_rows and not skipped_files:
            return ToolResult(success=False, message="Unable to read any table data")
        # 4. Semantic retrieval
        results = []
        if all_rows:
            texts = [f"Table name {r['table']}, composed of the first 2 rows of the table + data rows from a specified length:\n" + r["text"] for r in all_rows]
            if len(texts) >= 20000:
                return ToolResult(success=False, message="More than 10,000 text blocks in the retrieval path, cannot perform retrieval. Please narrow down the scope or use other tools.")
            default_batch_size = get_dynamic_batch_size(texts)
            results_sim = semantic_search(f"Task: You need to retrieve the corresponding rows in the table based on the query: {query}", texts, top_k, cache_path=cache_path, batch_size=default_batch_size)
            # 5. Build results
            for r in results_sim:
                item = all_rows[r.index]
                results.append({
                    "table": item["table"], 
                    "header": item["header"], 
                    "row": item["row"], 
                    "text": item["text"],
                    "score": float(r.score)
                })
        return self.make_result(
            success=True,
            data={"results": results, "skipped_files": [str(f) for f in skipped_files]},
            message=f"Retrieved {len(results)} results from {len(all_rows)} rows in {len(processing_files)} files"
        )
    
    def format_output(self, data) -> str:
        """Format retrieval results as Markdown table"""
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            skipped_files = data.get("skipped_files", [])
        else:
            results = data
            skipped_files = []
        output_lines = []
        if not results:
            output_lines.append("No retrieval results")
        else:
            # Group by source file
            from collections import defaultdict
            groups = defaultdict(list)
            for item in results:
                groups[item["table"]].append(item)
            for table, items in groups.items():
                display_path = table
                output_lines.append(f"**Source: {display_path}**\n")
                
                for item in items:
                    # Try to parse line number range
                    start_row = item.get("row", "?")
                    score = item.get("score", 0.0)
                    text_content = item.get("text", "")
                    
                    # Try to calculate end line number and clean up display content
                    display_text = text_content
                    end_row = "?"
                    
                    try:
                        # Assume text contains "Specific table data:\n" separator (wait, I translated it earlier)
                        if "Specific table data:\n" in text_content:
                            parts = text_content.split("Specific table data:\n")
                            # parts[0] is header section
                            # parts[1] is data section
                            
                            header_part = parts[0]
                            data_part = parts[1]
                            
                            # Calculate line count
                            line_count = len(data_part.strip().splitlines())
                            if isinstance(start_row, int):
                                end_row = start_row + line_count - 1
                                
                            # Clean up header part for display
                            # It is constructed as f"Header data:\n{header_info}\n-------\n"
                            clean_header = header_part.replace("Header data:\n", "").replace("\n-------\n", "").strip()
                            
                            display_text = f"{clean_header}\n{data_part.strip()}"
                    except:
                        pass
                        
                    output_lines.append(f"> Relevance: {score:.4f} | Line range: {start_row} - {end_row}")
                    output_lines.append("```csv")
                    output_lines.append(display_text.strip())
                    output_lines.append("```\n")
                    
        if skipped_files:
            output_lines.append(f"\nWarning: There are {len(skipped_files)} xlsx or xls table files. Currently, only retrieval for csv and tsv formats is supported.")
            if len(skipped_files) > 3:
                show_files = skipped_files[:3]
                output_lines.append("Paths for first 3 files:")
            else:
                show_files = skipped_files
                output_lines.append("File paths:")
            for f in show_files:
                rel_path = f
                output_lines.append(f"- {rel_path}")
        return "\n".join(output_lines)
