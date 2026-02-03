"""
Table Localization Tool
Scans folders and locates table files most relevant to the query
"""
from typing import List, Dict, Any
from pathlib import Path

from src.tools.base import BaseTool, ToolResult, register_tool
from src.utils.table_process import scan_table_files, get_all_sheets_preview
from src.retrival import semantic_search
from src.utils.global_config import dataset_list
from src.utils.common import get_dynamic_batch_size
import os
import hashlib
import pickle
from src.utils.chat_api import ChatClient

def build_folder_tree(table_files: List[Dict[str, Any]], max_files: int = 10) -> str:
    """
    Build a tree structure display for the folder
    
    Rules:
    1. Prioritize displaying the outermost structure (all files and folders under the root directory)
    2. If display count is less than max_files, try to recursively expand the first folder (depth priority)
    3. Ensure the outermost structure is displayed as completely as possible
    
    Args:
        table_files: List of table file information
        max_files: Maximum number of files/folders to display
        
    Returns:
        Tree structure string
    """
    # 1. Build tree structure
    # Node structure: {'type': 'dir'|'file', 'name': str, 'children': {name: node}}
    root = {'type': 'dir', 'name': '', 'children': {}}
    
    for tf in table_files:
        rel_path = tf['relative_path']
        parts = Path(rel_path).parts
        
        current = root
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # File
                if part not in current['children']:
                    current['children'][part] = {'type': 'file', 'name': part, 'children': {}}
            else:
                # Directory
                if part not in current['children']:
                    current['children'][part] = {'type': 'dir', 'name': part, 'children': {}}
                current = current['children'][part]

    # 2. Determine nodes to display
    # Sorting helper function: directories first, then alphabetical order
    def get_sorted_children(node):
        return sorted(node['children'].values(), key=lambda x: (0 if x['type'] == 'dir' else 1, x['name']))

    root_children = get_sorted_children(root)
    visible_paths = set()
    current_quota = max_files
    
    # Strategy:
    # First level: display as many sub-items in the root directory as possible
    root_children_paths = [child['name'] for child in root_children]
    to_show_root = root_children_paths[:current_quota]
    visible_paths.update(to_show_root)
    current_quota -= len(to_show_root)
    
    # If quota remains, try to recursively expand the first directory
    def expand_recursive(node, current_path_parts):
        nonlocal current_quota
        if current_quota <= 0:
            return

        children = get_sorted_children(node)
        if not children:
            return
            
        children_paths = []
        for child in children:
            # Join paths using /
            child_path = "/".join(current_path_parts + [child['name']])
            children_paths.append(child_path)
            
        # If the entire current level can be displayed
        if len(children) <= current_quota:
            visible_paths.update(children_paths)
            current_quota -= len(children)
            
            # Continue trying to expand the first directory in this level
            for child in children:
                if child['type'] == 'dir':
                    expand_recursive(child, current_path_parts + [child['name']])
                    break # Expand only the first branch
        else:
            # Only partial display possible
            to_show = children_paths[:current_quota]
            visible_paths.update(to_show)
            current_quota = 0
            return

    # Find the first visible directory under the root directory to expand
    first_dir = None
    for node in root_children:
        if node['name'] in visible_paths and node['type'] == 'dir':
            first_dir = node
            break
            
    if first_dir:
        expand_recursive(first_dir, [first_dir['name']])

    # 3. Generate output string
    lines = []
    
    def render(node, prefix, is_last, path_parts):
        name = node['name']
        path = "/".join(path_parts)
        
        connector = "└── " if is_last else "├── "
        
        if node['type'] == 'dir':
            icon = "📁 "
            display_name = name + "/"
        else:
            icon = "📄 "
            display_name = name
            
        lines.append(f"{prefix}{connector}{icon}{display_name}")
        
        if node['type'] == 'dir':
            children = get_sorted_children(node)
            visible_children = [c for c in children if "/".join(path_parts + [c['name']]) in visible_paths]
            
            child_prefix = prefix + ("    " if is_last else "│   ")
            
            if visible_children:
                for i, child in enumerate(visible_children):
                    # Check if there are more undisplayed children
                    has_more = len(visible_children) < len(children)
                    is_last_child = (i == len(visible_children) - 1)
                    
                    if has_more:
                        # If there are undisplayed items, use ├── for the last visible child
                        render(child, child_prefix, False, path_parts + [child['name']])
                    else:
                        render(child, child_prefix, is_last_child, path_parts + [child['name']])
                
                if len(visible_children) < len(children):
                    omitted_count = len(children) - len(visible_children)
                    lines.append(f"{child_prefix}└── ... ({omitted_count} files omitted)")
            elif children:
                # If no visible children but directory contains files, show ellipsis
                lines.append(f"{child_prefix}└── ...")

    # Render root directory
    root_visible_children = [c for c in root_children if c['name'] in visible_paths]
    for i, child in enumerate(root_visible_children):
        has_more = len(root_visible_children) < len(root_children)
        is_last = (i == len(root_visible_children) - 1) and not has_more
        render(child, "", is_last, [child['name']])
        
    if len(root_visible_children) < len(root_children):
        omitted = len(root_children) - len(root_visible_children)
        lines.append(f"└── ... ({omitted} files omitted)")
    
    # Count total undisplayed files
    shown_files_count = 0
    for tf in table_files:
        if tf['relative_path'] in visible_paths:
            shown_files_count += 1
            
    total_files = len(table_files)
    if shown_files_count < total_files:
        lines.append(f"... ({total_files - shown_files_count} more files not shown)")
    
    return "\n".join(lines)


# @register_tool(use_cache=True)
class TableLocator(BaseTool):
    """
    Table Localization Tool
    Scans specified folder, lists total table count and project structure overview
    """
    name = "table_locator_backup"
    ### description = "Type: File Localization. Scans folder for table files, returns total count and brief project structure. Used for understanding data file distribution."
    description = "Type: File Localization. Precise path scanning. Similar to os.listdir but with hierarchical structure display.\n    *   **Applicable Scenario**: Use when you need to quickly understand the distribution, quantity, and hierarchical structure of files under a folder.\n    *   **Example**: List all subfolders and files under the dataset directory."
    category = "location"

    parameters = {
        "folder_path": {
            "type": "string",
            "description": "Absolute path of the folder to scan",
            "required": True
        },
        "max_show_files": {
            "type": "integer",
            "description": "Maximum number of tables to display under this path, default is 10",
            "required": False
        }
    }
    
    def execute(self, folder_path: str = None, max_show_files: int = 10, **kwargs) -> ToolResult:
        """Execute table localization"""
        assert folder_path is not None, "Please provide folder_path parameter"
        
        folder_path = os.path.abspath(folder_path)
        
        # Scan table files
        table_files = scan_table_files(folder_path)
        if not table_files:
            return ToolResult(success=False, message=f"No table files found in folder '{folder_path}' (supported: csv, xlsx, xls, tsv)")
        # Statistics
        total_count = len(table_files)
        # Build output
        output_lines = [
            f"## Folder Overview", "", f"**Folder**: `{folder_path}`",
            f"**Total Files**: {total_count}", "", f"## 📁 Project Structure", "```",
            f"{Path(folder_path).name}/", build_folder_tree(table_files, max_files=max_show_files),
            f"```",
            "",
            "**Note**: Outer structures and partial folders are prioritized. Omitted items may also contain sub-files. For more details, call this tool with a more precise path or use other tools."
        ]
        
        return ToolResult(
            success=True, data="\n".join(output_lines),
            message=f"Found {total_count} table files"
        )


# @register_tool(use_cache=True)
class TableLocator(BaseTool):
    """
    Table Localization Tool
    Scans specified folder and lists all files
    """
    name = "table_locator"
    ### description = "Type: File Localization. Scans folder, returns file list. Used for understanding file distribution."
    description = "Type: Precise File Localization. Targeted folder directory preview. Lists all filenames under the specified folder.\n    *   **Applicable Scenario**: Similar to the terminal 'ls' command, used to confirm which files exist in the current directory.\n    *   **Example**: View the list of files under /tmp/data."
    category = "location"

    parameters = {
        "folder_path": {
            "type": "string",
            "description": "Absolute path of the folder to scan",
            "required": True
        }
    }
    
    def execute(self, folder_path: str = None, **kwargs) -> ToolResult:
        """Execute table localization"""
        assert folder_path is not None, "Please provide folder_path parameter"
        
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
             return ToolResult(success=False, message=f"Path does not exist: {folder_path}")
        try:
            # Scan table files for total count (reusing logic from backup)
            table_files = scan_table_files(folder_path)
            total_count = len(table_files) if table_files else 0

            # Get file list in current directory
            files = os.listdir(folder_path)
            files_str = "\n".join(files)
            return ToolResult(
                success=True, 
                data=f"Folder: {folder_path}\nTotal Files: {total_count} table files\nFiles in current directory:\n{files_str}",
                message=f"Found {total_count} table files"
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Error scanning folder: {str(e)}")


@register_tool()
class TableSelector(BaseTool):
    """
    Table Semantic Retriever
    Retrieves the most relevant tables based on semantic vectors in a folder containing numerous tables
    """
    
    name = "table_selector"
    ### description = "Type: File Localization. Semantic vector-based table retrieval. This semantic query is primarily based on semantic matching between actual table preview content and the query. It is suitable when there are many tables or when table names/path info are missing/insufficient for directory browsing."
    description = "Type: Semantic Retrieval. Target table localization via semantic retrieval. This semantic query matches actual table preview content with the provided query.\n **Applicable Scenarios**:\n1. Fast retrieval when there are many files.\n2. When table names, paths, and other external info cannot locate the table, and identifying it requires specific content. For example: non-standard filenames, filenames that don't reflect content, or searching for tables containing specific business meanings (e.g., 'sales revenue', 'Q1 financial report').\n3. Rich semantic info in the query improves retrieval accuracy. For example, query='Table containing statistics for first-quarter sales revenue' is better than simple 'sales'."
    category = "semantic_search"
    cache_file = "embeddings_cache.pkl"
    parameters = {
        "query": {
            "type": "string",
            ### "description": "Query description for semantic matching. Suggest generating multiple keywords or full sentences with rich information. Richer information helps in accurately locating the table based on content.",
            "description": "Query description. **Strongly recommend using complete descriptive sentences** instead of single keywords to improve semantic matching accuracy.",
            "required": True
        },
        "folder_path": {
            "type": "string",
            "description": "Absolute path of the table folder to retrieve from",
            "required": True
        },
        "top_k": {
            "type": "integer",
            "description": "Number of tables to return, default is 10",
            "required": False
        }
    }
    
    def execute(self, query: str, folder_path: str, top_k: int = 10, **kwargs) -> ToolResult:
        """Execute table semantic retrieval with caching mechanism"""
        folder_path = os.path.abspath(folder_path)
        # Determine base cache directory
        cache_base_dir = "./tmp"
        # Try to match dataset path
        matched_datasets = [item for item in dataset_list if item in folder_path]
        if matched_datasets:
            # If it belongs to a known dataset, store cache in dataset directory
            cache_base_dir = matched_datasets[0]
        os.makedirs(os.path.join(cache_base_dir, "tmp"), exist_ok=True)
        
        # Scan tables in folder
        table_files = scan_table_files(folder_path)
        if not table_files:
            return ToolResult(success=False, message=f"No table files found in folder '{folder_path}'")
        # V2 Update: Use _v2_relpath suffix and store relative paths to support cross-task cache reuse (different absolute paths but same relative path)
        path_hash = hashlib.md5((folder_path + f"{len(table_files)}_v5_relpath").encode('utf-8')).hexdigest()
        selector_cache_path = os.path.join(cache_base_dir, f"tmp/table_selector_meta_{path_hash}.pkl")
        embedding_cache_path = os.path.join(cache_base_dir, self.cache_file)
        if os.path.exists(selector_cache_path):
            try:
                with open(selector_cache_path, 'rb') as f:
                    table_docs, valid_tables = pickle.load(f)
                if not isinstance(table_docs, list) or not isinstance(valid_tables, list):
                    print(f"Warning: Cache format error in {selector_cache_path}, recreating...")
                    table_docs, valid_tables = None, None
            except Exception as e:
                print(f"Error loading cache {selector_cache_path}: {e}")
                table_docs, valid_tables = None, None
        else:
            table_docs, valid_tables = None, None

        if table_docs is None or valid_tables is None:
            table_docs, valid_tables = [], []
            for tf in table_files:
                previews = get_all_sheets_preview(tf['path'], max_rows=5)
                for preview in previews:
                    if not preview.get('error'):
                        # Build table description document
                        content_preview = '\n'.join(preview.get('head_lines', [])[:5])
                        file_name = preview['file']
                        sheet_name = preview.get('sheet', '')
                        if sheet_name:
                            display_name = f"{file_name}-[sheet: {sheet_name}]"
                        else:
                            display_name = file_name
                        
                        # Key modification: use relative path to build documents and store info
                        # This ensures the generated doc is consistent even if folder_path changes (e.g., in different temp dirs for different tasks), thus hitting the embedding cache.
                        rel_path = tf['relative_path']
                        preview['path'] = rel_path  # Store relative path
            
                        # Limit content preview length to prevent tokenizer from generating overly long sequences leading to OOM
                        tokenizer = ChatClient._get_tokenizer()
                        if tokenizer:
                            tokens = tokenizer.encode(content_preview)
                            if len(tokens) > 8000:
                                content_preview = tokenizer.decode(tokens[:8000]) + "...(truncated)"
                        elif len(content_preview) > 20000:
                            content_preview = content_preview[:20000] + "...(truncated)"
                            
                        doc = f"Table Name: {display_name}\nTable Path: {rel_path}\nTable Content Preview:\n{content_preview}"
                        table_docs.append(doc)
                        preview['display_name'] = display_name
                        valid_tables.append(preview)
            # Write to metadata cache
            with open(selector_cache_path, 'wb') as f:
                pickle.dump((table_docs, valid_tables), f)

        if not table_docs:
            return ToolResult(success=False, message="Unable to read any table files")
        # Build query
        search_query = f"Task: Based on the query: {query}, please retrieve the table(s) most likely to answer this question."
        top_k = min(top_k, len(table_docs))
        batch_size = get_dynamic_batch_size(table_docs)
        results = semantic_search(search_query, table_docs, top_k=top_k, cache_path=embedding_cache_path, batch_size=batch_size)
        matched_paths = []
        for r in results:
            if r.index < len(valid_tables):
                item = valid_tables[r.index]
                if isinstance(item, dict):
                    # item['path'] is now a relative path
                    rel_path_in_cache = item.get('path', '')
                    # Combine into the current absolute path
                    abs_path = os.path.abspath(os.path.join(folder_path, rel_path_in_cache))
                    display_name = item.get('display_name', Path(abs_path).name)
                    matched_paths.append(f"{display_name} (Path: {abs_path}) (Similarity: {r.score:.4f})")
        
        # Return concise list of paths
        return self.make_result(
            success=True,
            data="\n".join(matched_paths),
            message=f"Retrieved {len(matched_paths)} relevant tables: {', '.join([p.split(' (')[0] for p in matched_paths])}"
        )
