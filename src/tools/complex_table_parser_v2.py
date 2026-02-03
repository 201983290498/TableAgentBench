"""
Complex Table Parsing Tool V2
Automatically identify and process two types of complex CSV table structures:
- Type A: Complex Single Table - Contains hierarchical structure or merged cells
- Type B: Multi-Line Header Single Table - Header occupies multiple lines
"""
from typing import List, Dict, Any, Optional, Generator, Union
from pathlib import Path
import pandas as pd
import os
import sys

from src.tools.base import BaseTool, ToolResult, register_tool
from src.tools.header_merger import HeaderMerger

from src.function_llm.base_llm import BaseLLM
from src.utils.chat_api import ChatClient, get_chat_client
from src.utils.common import parse_json_response
from src.utils.table_process import get_table_preview_str
from src.prompts.ComplexTablePrompt import (
    COMPLEX_TABLE_HEADER_PARSER_PROMPT,
    COMPLEX_TABLE_TRANSFORM_SYSTEM,
)

COMPLEX_TABLE_CLASSIFIER_PROMPT = """# Task Requirements
You are an expert in table structure analysis. Determine the type of the given CSV snippet based on its structure and provide your reasoning to facilitate further table splitting.

# Type A: "Complex Single Table"
**Definition**: Essentially a single table, but in addition to column information in the header, there is a hierarchical structure in data rows with many merged cells (vertical/horizontal) that should have been part of the column header.
**Example**:
```plaintext
Region,2020_GDP,2021_GDP,2022_GDP,2023_GDP
Beijing,Beijing,Beijing,Beijing,Beijing
Haidian,10000,10500,11000,11500
Chaoyang,15000,15500,16000,16500
Fengtai,5000,5200,5400,5600
Shanghai,Shanghai,Shanghai,Shanghai,Shanghai
Pudong,20000,20500,21000,21500
Xuhui,10000,10200,10400,10600
Changning,10000,10300,10600,10900
```
> Example Explanation: This is a typical phenomenon when merged cells are converted to CSV. The 'Shanghai' and 'Beijing' rows are artifacts of splitting merged cells. The table describes economic data for Beijing and Shanghai regions. In practice, a 'City' column should be added, converting Beijing and Shanghai into column information.

# Type B: "Multi-Line Header Single Table"
**Definition**: Still a single table, but the header section is a multi-line hierarchical structure that needs to be merged into a single row. You need to identify how many rows the header occupies. Note: A multi-line header must satisfy the condition that all data rows below the columns are attributes of the corresponding columns. If this condition is not met, it cannot be merged as a multi-line header.
**Example**:
```plaintext
Region,Q1,,Q2,
,Sales (10k),Profit (10k),Sales (10k),Profit (10k)
Beijing,150.0,45.0,180.0,54.0
Shanghai,170.0,51.0,200.0,60.0
Guangzhou,130.0,39.0,160.0,48.0
Shenzhen,145.0,43.5,175.0,52.5
```
> Example Explanation: The flattened header should be "Region, Q1_Sales (10k), Q1_Profit (10k), Q2_Sales (10k), Q2_Profit (10k)"

# Input Data
{csv_content}

# Output Format
If Type A:
```json
{{"reason": "Detailed reasoning for determination", "type": "Type A"}}
```

If Type B:
Return reasoning + type + number of header rows.
```json
{{
    "reason": "Detailed reasoning",
    "type": "Type B",
    "header_rows": x
}}
```
"""

# ==================== LLM Helper Classes ====================

class _TableClassifier(BaseLLM):
    """Table type classifier"""
    def __init__(self, client: Optional[ChatClient] = None):
        super().__init__(client, system_prompt="You are an expert in table structure analysis.")
    
    def __call__(self, csv_content: str, **kwargs) -> Dict[str, Any]:
        prompt = COMPLEX_TABLE_CLASSIFIER_PROMPT.format(csv_content=csv_content)
        resp = self.client.batch_chat(
            prompts=[prompt],
            **kwargs
        )[0]
        result = parse_json_response(resp["content"])
        return result if isinstance(result, dict) else {"type": "Unknown", "error": str(result)}


class _HeaderParser(BaseLLM):
    """Complex table header parser"""
    def __init__(self, client: Optional[ChatClient] = None):
        super().__init__(client, system_prompt="You are an expert in complex report parsing.")
    
    def __call__(self, csv_content: str) -> Dict[str, Any]:
        prompt = COMPLEX_TABLE_HEADER_PARSER_PROMPT.format(csv_content=csv_content)
        resp = self.client.batch_chat(
            prompts=[prompt],
            enable_thinking=True,
        )[0]
        result = parse_json_response(resp["content"])
        return result if isinstance(result, dict) else {"error": str(result)}


class _DataTransformer(BaseLLM):
    """Data transformer (multi-turn conversation)"""
    def __init__(self, new_columns: List[str], extraction_rules: str, client: Optional[ChatClient] = None):
        system = COMPLEX_TABLE_TRANSFORM_SYSTEM.format(new_columns=new_columns, extraction_rules=extraction_rules)
        super().__init__(client, system_prompt=system)
        self.new_columns = new_columns
        self.history: List[Dict[str, str]] = []
    
    def __call__(self, csv_chunk: str) -> List[List[str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.history:
            messages.append(self.history[-2])  # Previous user input
            messages.append(self.history[-1])  # Previous assistant reply
        messages.append({"role": "user", "content": csv_chunk})
        
        resp = self.client.batch_chat(
            messages=[messages],
            enable_thinking=True
        )[0]
        self.history.append({"role": "user", "content": csv_chunk})
        self.history.append({"role": "assistant", "content": resp["content"]})
        result = parse_json_response(resp["content"])
        return result.get("data_items", []) if isinstance(result, dict) else []


# ==================== Tool Class ====================

@register_tool()
class ComplexTableParserV2(BaseTool):
    """
    Complex Table Parsing Tool
    
    Function: Automatically identify CSV table structure types and convert them into standard 1D relational tables.
    
    Applicable Scenarios (When to use):
    - When ordinary pd.read_csv fails or the resulting data structure is messy.
    - When the table contains multi-level headers, merged cells, or appears to be a table visually but has complex structure.
    
    Supports processing the following table types:
    - Type A (Complex Single Table): Complex structured single table (contains layers/merged cells), will be automatically flattened into a standard 1D table.
    - Type B (Multi-Line Header Single Table): Multi-level header single table, will automatically merge headers into one line.
    
    Note: Only supports CSV files. For Excel files (.xlsx/.xls), please use xlsx_to_csv_converter first.
    """
    
    name = "complex_table_parser_v2"
    description = """Type: Table Understanding. Specialized processing tool for extra-large (over 50 rows, 20 columns) non-standard tables.
Key features:
1.【Table Structure Reconstruction】(Type A): Suitable for tables where data rows contain hierarchical structures or many merged cells (vertical/horizontal) besides header column information. Converts these tables into standard 2D tables.
2.【Header Flattening】(Type B): Flattens complex multi-level headers into a 1D relational table;
> Note: Only suitable for large tables."""
    category = "understanding"
    
    parameters = {
        "csv_path": {
            "type": "string",
            "description": "Absolute path to the CSV file (only .csv format supported). For xlsx files, first convert them to csv.",
            "required": True
        },
        "output_dir": {
            "type": "string",
            "description": "Absolute path to the output directory, defaults to creating one under the source file directory",
            "required": False
        }
    }
    
    def __init__(self):
        self.client = None
        self.classifier = None
        self.header_merger = HeaderMerger()
    
    def _ensure_client(self):
        if self.client is None:
            self.client = get_chat_client()
            self.classifier = _TableClassifier(self.client)
    
    def execute(self, csv_path: str, output_dir: str = None) -> ToolResult:
        """
        Execute complex table parsing
        
        Args:
            csv_path: CSV file path
            output_dir: Output directory
            
        Returns:
            ToolResult: Contains table type and output file path
        """
        try:
            self._ensure_client()
            
            # Validate file
            _path = Path(os.path.abspath(csv_path))
            if not _path.exists():
                return ToolResult(False, message=f"File does not exist: {csv_path}", data=f"File does not exist: {csv_path}")
            if not _path.suffix.lower() == '.csv':
                return ToolResult(False, message="Only CSV files are supported, please convert xlsx to csv first", data="Only CSV files are supported, please convert xlsx to csv first")
            
            # Output directory
            _output = Path(os.path.abspath(output_dir)) if output_dir else _path.parent / f"{_path.stem}_parsed"
            _output.mkdir(parents=True, exist_ok=True)
            
            # Classification
            csv_content = self._read_preview(_path)
            classify_result = self.classifier(csv_content, enable_thinking=True)
            table_type = classify_result.get("type", "Unknown")
            
            # Processing
            if table_type == "Type A":
                # Type A: Complex Single Table
                output_info = self._handle_type_a(_path, _output)
            elif table_type == "Type B":
                # Type B: Multi Lines Header Single Table
                header_rows = classify_result.get("header_rows", 2)
                output_info = self._handle_type_b(_path, _output, header_rows)
            else:
                return ToolResult(False, message=f"Unrecognized or unsupported table type: {table_type}", data=f"Unrecognized or unsupported table type: {table_type}")
            
            # Concise output
            output_files = self._extract_output_files(output_info)
            
            # Generate preview
            preview_lines = []
            for f in output_files:
                 preview = get_table_preview_str(f, n=10)
                 preview_lines.append(f"File: {f}\n{preview}")

            summary = f"Type: {table_type}\nOutput Files:\n" + "\n".join(f"  - {f}" for f in output_files)
            if preview_lines:
                summary += "\n\nPreview:\n" + "\n\n".join(preview_lines)

            if "error" in output_info:
                summary += f"\nError Message: {output_info['error']}"
            return ToolResult(True, data=summary, ori_data={"type": table_type, "files": output_files, "error": output_info.get("error")}, message=summary)
        except Exception as e:
            return ToolResult(False, message=f"Parsing failed: {str(e)}", data=f"Parsing failed: {str(e)}")
    
    def _read_preview(self, path: Path, max_lines: int = 100) -> str:
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return '\n'.join(line.rstrip() for i, line in enumerate(f) if i < max_lines)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to read file: {path}")
    
    def _read_chunks(self, path: Path, chunk_size: int = 200) -> Generator[str, None, None]:
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                for i in range(0, len(lines), chunk_size):
                    yield ''.join(lines[i:i + chunk_size])
                return
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to read file: {path}")
    
    def _handle_type_a(self, csv_path: Path, output_dir: Path) -> Dict:
        """Handle Type A: Complex Single Table (Flattening)"""
        preview = self._read_preview(csv_path)
        header_parser = _HeaderParser(self.client)
        header_info = header_parser(preview)
        
        new_columns = header_info.get("new_columns", [])
        extraction_rules = header_info.get("extraction_rules", "")
        if not new_columns:
            return {"error": "Failed to parse header", "files": []}
        
        transformer = _DataTransformer(new_columns, extraction_rules, self.client)
        all_data = []
        for chunk in self._read_chunks(csv_path):
            all_data.extend(transformer(chunk))
        
        if not all_data:
            return {"error": "No valid data converted", "files": []}
        
        # Align column count
        valid_data = []
        for row in all_data:
            if isinstance(row, list):
                if len(row) < len(new_columns):
                    row = row + [''] * (len(new_columns) - len(row))
                valid_data.append(row[:len(new_columns)])
        
        out_path = output_dir / f"{csv_path.stem}_flattened.csv"
        pd.DataFrame(valid_data, columns=new_columns).to_csv(out_path, index=False, encoding='utf-8-sig')
        return {"files": [str(out_path)]}
    
    def _handle_type_b(self, csv_path: Path, output_dir: Path, header_rows: int) -> Dict:
        """Handle Type B: Multi Lines Header (Header Merging)"""
        result = self.header_merger.execute(str(csv_path), header_rows, str(output_dir / f"{csv_path.stem}_merged.csv"))
        if result.success:
            return {"files": [result.ori_data.get("Output File")]}
        else:
            return {"error": result.message, "files": []}
    
    def _extract_output_files(self, output_info: Dict) -> List[str]:
        if "files" in output_info:
            return output_info["files"]
        if "sub_tables" in output_info:
            return [s.get("file", "") for s in output_info["sub_tables"] if s.get("file")]
        if "result" in output_info and isinstance(output_info["result"], dict):
            result = output_info["result"]
            if "Split results" in result:
                return [s.get("file", "") for s in result["Split results"]]
        return []
