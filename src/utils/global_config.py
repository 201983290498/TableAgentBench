GLOBAL_CONFIG = {
    "table_agent": {
        "max_llm_tokens": "20480",
        "max_tool_response": "6144",
    },
    "tools": {
        "grep_search": {
            "max_detail_files": 5,  # Number of files to display in detail
            "max_matches_per_file": 10,  # Maximum number of matches to display per file
            "default_fine_grained_matches_to_show": 10  # Default number of maximum matches returned by fine-grained retrieval mode.
        }
    }
}

dataset_list = ["dataset"]
