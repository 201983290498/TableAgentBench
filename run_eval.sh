# python evaluate.py \
#     --mode all \
#     --user_model "qwen3-coder-plus" \
#     --config_key "qwen3-coder-plus" \
#     --eval_file "dataset/通用/samples.json" \
#     --table_path "dataset/通用/表格数据" \
#     --workers 1 \
#     --max_steps 70 \
#     --max_history_tokens 65534 \
#     --auto_parse True \
#     --enable_thinking True \
#     --eval_config_key "deepseek-deepseek-v3.2" \
#     --unimodel "deepseek-deepseek-v3.2" \
#     --verbose True \
#     --trace_save_dir "traces_output/all_tools"


python evaluate.py \
    --mode all \
    --user_model "qwen3-coder-plus" \
    --config_key "qwen3-coder-plus" \
    --eval_file "dataset/通用/samples.json" \
    --table_path "dataset/通用/预处理后的表" \
    --workers 1 \
    --max_steps 70 \
    --max_history_tokens 65534 \
    --auto_parse True \
    --enable_thinking True \
    --eval_config_key "deepseek-deepseek-v3.2" \
    --unimodel "deepseek-deepseek-v3.2" \
    --verbose True \
    --include_tools "python_code_executor,cmd_executor,grep_search,table_selector,semantic_column_retriever,semantic_row_retriever" \
    --trace_save_dir "traces_output/wo_process-pre-qwen3" > log.log

    
python evaluate.py \
    --mode all \
    --user_model "kimi-k2-0905" \
    --config_key "kimi-k2-0905" \
    --eval_file "dataset/通用/samples.json" \
    --table_path "dataset/通用/预处理后的表" \
    --workers 1 \
    --max_steps 70 \
    --max_history_tokens 65534 \
    --auto_parse True \
    --enable_thinking True \
    --eval_config_key "deepseek-deepseek-v3.2" \
    --unimodel "deepseek-deepseek-v3.2" \
    --verbose True \
    --include_tools "python_code_executor,cmd_executor,grep_search,table_selector,semantic_column_retriever,semantic_row_retriever" \
    --trace_save_dir "traces_output/wo_process-pre-kimi" > log.log
