"""
Semantic Column Retrieval Tool
Recalls the most relevant columns in table files based on vector similarity to the query
Supported formats: csv, xlsx, xls, tsv
"""
from typing import List
from pathlib import Path
import os
from src.tools.base import BaseTool, ToolResult, register_tool
from src.retrival.embedder import semantic_search
from src.utils.global_config import dataset_list
from src.utils.table_process import scan_table_files, read_table_lines

@register_tool()
class SemanticColumnRetriever(BaseTool):
    """
    Semantic Column Retrieval Tool
    Recalls the most relevant columns in table files based on embedding similarity. Can only handle simple tables, does not support complex tables (e.g., merged cells, multi-level headers).
    Supported: csv, xlsx, xls, tsv
    """
    
    name = "semantic_column_retriever"
    ### description = "Type: Content Retrieval. Perform semantic search in specified table files or folders to recall the most relevant table columns to the query. Supports csv, tsv formats. Please output complete query for column recall."
    description = """Type: Semantic Retrieval. Based on semantic retrieval to find relevant table columns under specified files or folders for a given query. It uses column names and sample values for semantic matching. Different from table location, it enables finer-grained column positioning.
Applicable Scenarios:
• Quickly locate target fields in ultra-wide tables (50+ columns)
• When table column names use professional abbreviations or codes that differ from user terminology, semantic retrieval is needed to find the most relevant columns in specified files or folders.
• When user queries for indicators cannot be found directly through keyword searching in column names.
**Example**: Input query='find column representing user age', can match 'u_age' or 'user_years'."""
    category = "semantic_search"
    cache_file = "column_embeddings_cache.pkl"

    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.tsv']
    
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute path of table file or folder (supports csv, tsv)",
            "required": True
        },
        "query": {
            "type": "string",
            ### "description": "Query text, describing characteristics of columns to find, or specifically what query to match corresponding columns against",
            "description": "Query text, describing features of the column to find. It is recommended to use complete natural language descriptions, e.g., 'find columns representing sales amount'. Richer query information helps improve retrieval accuracy.",
            "required": True
        },
        "top_k": {
            "type": "integer",
            "description": "Number of columns to return, defaults to 10",
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
        """Load table and prepare column data, supports csv, xlsx, xls, tsv"""
        lines, _, total = read_table_lines(str(table_path), mode="all")
        if not lines:
            return []
        # First row is the header
        header = lines[0].split(',')
        columns_data = []
        # Collect sample values for each column (up to 5 non-empty values)
        for col_idx, col_name in enumerate(header):
            sample_values = []
            for line in lines[1:6]:  # Take first 5 rows of data as samples
                values = line.split(',')
                if col_idx < len(values) and values[col_idx].strip():
                    sample_values.append(values[col_idx])
            sample_str = ", ".join(sample_values)
            columns_data.append({
                "table": str(table_path),
                "column": col_name,
                "text": f"Table path: {str(table_path)} | Column name: {col_name} | Sample values: {sample_str}"
            })
        return columns_data
    
    def execute(self, path: str, query: str, top_k: int = 10, **kwargs) -> ToolResult:
        """Execute semantic column retrieval"""
        path = os.path.abspath(path)
        # 0. Try to load cache (for T2R dataset)
        if len([item for item in dataset_list if item in path])>0: 
            cache_path = os.path.join([item for item in dataset_list if item in path][0], self.cache_file)
        else:
            cache_path = None
        # 1. Collect table files
        table_files = self._collect_table_files(path)
        if not table_files:
            return ToolResult(success=False, message=f"No supported table files found: {path}")
        
        # Distinguish between processing files and skipped ones
        processing_files = []
        skipped_files = []
        for f in table_files:
            if f.suffix.lower() in ['.csv', '.tsv']:
                processing_files.append(f)
            elif f.suffix.lower() in ['.xlsx', '.xls']:
                skipped_files.append(f)

        # 2. Load all column data
        all_columns = []
        for table_file in processing_files:
            try:
                cols = self._load_and_prepare(table_file)
                all_columns.extend(cols)
            except Exception as e:
                continue  # Skip files that fail to read
        
        if not all_columns and not skipped_files:
            return ToolResult(success=False, message="Unable to read any table data")
        
        # 3. Semantic retrieval
        results = []
        if all_columns:
            texts = ["Table column information: " + c["text"] for c in all_columns]
            if len(texts) >= 10000:
                return ToolResult(success=False, message="More than 10,000 columns in search path, unable to recall. Please narrow the search scope or use other tools.")
            results_sim = semantic_search(f"Task: You need to retrieve the corresponding columns from the table based on the query: {query}", texts, top_k, cache_path=cache_path)
            # 4. Construct results
            for r in results_sim:
                item = all_columns[r.index]
                results.append({
                    "table": item["table"],
                    "column": item["column"],
                    "score": float(r.score)
                })

        return self.make_result(
            success=True,
            data={"results": results, "skipped_files": [str(f) for f in skipped_files]},
            message=f"Recalled {len(results)} results from {len(all_columns)} columns in {len(processing_files)} files"
        )

    def format_output(self, data) -> str:
        """Format recall results"""
        # Handle the new data structure or legacy list
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            skipped_files = data.get("skipped_files", [])
        else:
            results = data
            skipped_files = []

        output_lines = []
        
        if not results:
            output_lines.append("No recall results")
        else:
            # Group by source file
            from collections import defaultdict
            groups = defaultdict(list)
            for item in results:
                groups[item["table"]].append(item["column"])
            
            import os
            for table, columns in groups.items():
                # Try to get path relative to current working directory
                display_path = table
                output_lines.append(f"File path: {display_path}")
                output_lines.append(f"Recalled columns: {', '.join(columns)}")
                output_lines.append("=" * 30)
        
        # Add warning for skipped files
        if skipped_files:
            output_lines.append(f"\nWarning: There are {len(skipped_files)} xlsx or xls table files. Currently, search and recall only support csv and tsv formats.")
            if len(skipped_files) > 3:
                show_files = skipped_files[:3]
                output_lines.append("The first 3 file paths are as follows:")
            else:
                show_files = skipped_files
                output_lines.append("File paths are as follows:")
            
            for f in show_files:
                rel_path = f
                output_lines.append(f"- {rel_path}")

        return "\n".join(output_lines)

