"""
Complex Header Merger Tool
Merges multi-line headers into a single row, connecting levels with '-'
"""
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import os

from src.tools.base import BaseTool, ToolResult, register_tool


@register_tool()
class HeaderMerger(BaseTool):
    """
    Complex Header Merger Tool
    
    Features:
    1. Identify multi-line headers in CSV files
    2. Merge multi-line headers into a single row, connecting levels with '-'
    3. Handle horizontal and vertical merged cells
    4. Output a new CSV file
    """
    
    name = "header_merger"
    description = """Type: Table Processing. Intelligently merges multi-line headers in CSV files into a single row, using '-' to represent hierarchical relationships.
**Applicable Scenarios**:
1. Flattening long multi-level headers for single tables, supporting subsequent behaviors like field retrieval, filtering, comparison, and aggregation.
2. Not suitable for CSV files containing multiple sub-tables.
3. Input: CSV file path and number of header rows; Output: Standardized CSV file."""

    category = "table_process"
    parameters = {
        "csv_path": {
            "type": "string",
            "description": "Full path to the CSV file",
            "required": True
        },
        "header_rows": {
            "type": "integer",
            "description": "Number of rows occupied by the complex header (first N rows), to be determined by the model based on file content",
            "required": True
        },
        "output_path": {
            "type": "string",
            "description": "Absolute path for the output file; if empty, '_merged' is appended to the original filename",
            "required": False
        }
    }
    
    def execute(self, csv_path: str, header_rows: int, output_path: str = None) -> ToolResult:
        """
        Execute header merging
        Args:
            csv_path: CSV file path
            header_rows: Number of header rows
            output_path: Output path
        Returns:
            ToolResult: Contains processing results
        """
        try:
            # 1. Validate input
            csv_file = Path(csv_path)
            if not csv_file.exists():
                msg = f"File does not exist: {csv_path}"
                return ToolResult(success=False, data=msg, message=msg)
            if header_rows < 1:
                msg = "Number of header rows must be at least 1"
                return ToolResult(success=False, data=msg, message=msg)
            # 2. Read CSV
            df = pd.read_csv(csv_path, header=None, keep_default_na=False)
            if len(df) < header_rows:
                msg = f"Total lines in file ({len(df)}) is less than specified header rows ({header_rows})"
                return ToolResult(success=False, data=msg, message=msg)
            # 3. If only 1 header row, return directly
            if header_rows == 1:
                output_path = output_path or str(csv_file.parent / f"{csv_file.stem}_merged.csv")
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                rel_path = os.path.abspath(output_path)
                return self.make_result(True, {
                    "Before Conversion": [df.iloc[0].tolist()],
                    "After Conversion": df.iloc[0].tolist(),
                    "Output File": rel_path
                }, f"Only one header row, no merging needed\nOutput file: {rel_path}")
            # 4. Get original header
            original_header = [df.iloc[i].tolist() for i in range(header_rows)]
            # 5. Merge header
            flattened_header = self._flatten_top_n_rows(df, header_rows)
            # 6. Construct new DataFrame
            header_df = pd.DataFrame([flattened_header])
            df.columns = range(len(df.columns))
            df_new = pd.concat([header_df, df.iloc[header_rows:]], ignore_index=True)
            # 7. Output file
            output_path = output_path or str(csv_file.parent / f"{csv_file.stem}_merged.csv")
            df_new.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')
            
            rel_path = os.path.abspath(output_path)
            from src.utils.table_process import get_table_preview_str
            preview = get_table_preview_str(rel_path, n=10)
            msg = f"Successfully merged {header_rows} header rows into one\nOriginal columns: {len(df.columns)}\nOutput file: {rel_path}\n\nPreview:\n{preview}"
            return self.make_result(True, {
                "Before Conversion": original_header,
                "After Conversion": flattened_header,
                "Output File": rel_path
            }, msg)
        except Exception as e:
            msg = f"Header merging failed: {str(e)}"
            return ToolResult(success=False, data=msg, message=msg)
    
    def _flatten_top_n_rows(self, df: pd.DataFrame, header_rows_num: int) -> List[str]:
        """
        Merge the first N rows into a single header row, concatenated with '-'
        
        Logic:
        1. Vertical merge: Empty cells inherit the nearest non-empty value from above
        2. Horizontal merge: Empty cells inherit the nearest non-empty value from the left
        3. Join levels with '-', deduplicate
        """
        df_head = df.iloc[:header_rows_num]
        
        # Extract header rows for each column
        column_headers = [
            [str(df_head.iat[row, col]).strip() if df_head.iat[row, col] else '' 
             for row in range(header_rows_num)]
            for col in range(len(df_head.columns))
        ]
        
        new_column_headers = []
        
        for col_idx, col_header in enumerate(column_headers):
            new_col_header = []
            cur_cell = ''  # Nearest non-empty cell from above
            
            for row_idx, cell in enumerate(col_header):
                if not cell:  # Fill empty cell
                    if cur_cell:  # Vertical inheritance: non-empty value exists above
                        col_header[row_idx] = cur_cell
                    else:  # Horizontal inheritance: look for non-empty value to the left
                        left_col_idx = col_idx - 1
                        while left_col_idx >= 0:
                            if column_headers[left_col_idx][row_idx]:
                                col_header[row_idx] = column_headers[left_col_idx][row_idx]
                                new_col_header.append(column_headers[left_col_idx][row_idx])
                                break
                            left_col_idx -= 1
                else:  # Add directly if non-empty
                    cur_cell = cell
                    new_col_header.append(cell)
            new_column_headers.append(new_col_header)
        # Concatenate levels with '-' and deduplicate
        flattened_header = ['-'.join(dict.fromkeys(filter(None, col))) for col in new_column_headers]
        return flattened_header
