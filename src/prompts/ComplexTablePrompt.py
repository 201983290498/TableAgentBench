"""
Complex Table Parsing Prompts
Used for table classification, header parsing, data transformation
"""

# Table Type Classification Prompt
COMPLEX_TABLE_CLASSIFIER_PROMPT = """# Task Requirements
You are a professional table structure analysis expert. Analyze the given CSV table data fragment and classify it into one of the following three types:

Type A: "Discrete Tables" (Discrete Multiple Tables)
- The file contains multiple independent tables, separated by empty rows or empty columns.
- The structure between tables may be different (different number of columns or misalignment).

Type B: "Fixed+Repeated" (Fixed Columns + Repeated Meaning Columns)
- Single table, the first few columns are "fixed columns", followed by repeated columns with similar meanings but slight differences.
- For example: 2021Q1 Sales, 2021Q2 Sales, 2022Q1 Sales...
- If it is this type, you need to identify fixed columns and repeated column groups, and split them into multiple sub-tables.

Type C: "Complex Table" (Complex Structure Table)
- There is a hierarchical structure in the data rows, with a large number of merged cell effects.
- Presented as a large number of duplicate values or data inconsistent with column meanings in CSV.

# Input Data
{csv_content}

# Output Format
If Type A or Type C:
```json
{{"reason": "Reason for judgment", "type": "Type A"}}
```
or
```json
{{"reason": "Reason for judgment", "type": "Type C"}}
```

If Type B:
```json
{{
    "reason": "Reason for judgment",
    "type": "Type B",
    "sub_tables": [
        {{"columns": ["Fixed Column 1", "Fixed Column 2", "Repeated Column 1", "Repeated Column 2"], "table_name": "Sub-table Name 1"}},
        {{"columns": ["Fixed Column 1", "Fixed Column 2", "Repeated Column 3", "Repeated Column 4"], "table_name": "Sub-table Name 2"}}
    ]
}}
```"""

# Complex Header Parsing Prompt
COMPLEX_TABLE_HEADER_PARSER_PROMPT = """You are a complex report parsing expert. Analyze complex tables and convert hierarchical structures into flat tables.

# Task
1. Identify the original headers and their meanings. The identified original headers need to remain unchanged in the new column names to facilitate data lookup.
2. Identify hierarchical structures in data rows (e.g., categories, subcategories, etc.).
3. Define new flattened headers, converting hierarchical information into independent columns. The new columns formed after conversion need to clearly reflect the column meanings.
4. Define how to extract new column values from data rows, and how to inherit information from "Header Rows" to "Detail Rows".

# Input Data (First 50 lines)
{csv_content}

# Output Format
```json
{{
    "new_columns": ["New Column Name 1", "New Column Name 2", ...],
    "extraction_rules": "Detailed description of column mapping relationships and data extraction rules, including how to handle hierarchy inheritance"
}}
```"""

# Data Transformation System Prompt
COMPLEX_TABLE_TRANSFORM_SYSTEM = """You are a data transformation engine. Flatten complex CSV tables into a one-dimensional table.

**Target Header**: {new_columns}
**Transformation Rules**: {extraction_rules}

The user will provide raw CSV data in batches. Please transform and output according to the rules.
- Only output valid data rows, skip pure header rows and empty rows.
- For hierarchical structures, inherit parent information to child rows.
- Ensure each row of data has complete column values.

Output Format:
```json
{{"data_items": [["Value 1", "Value 2", ...], ["Value 1", "Value 2", ...]]}}
```"""
