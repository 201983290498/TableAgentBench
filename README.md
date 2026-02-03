# TableAgent Evaluation Framework

English | [中文](README_zh.md)

This repository contains a framework for evaluating LLM-based agents on table-related tasks (Table QA). It is designed to assess the ability of agents to interact with structured data (Excel, CSV) to answer complex natural language queries.

## Features

- **Multi-Model Support**: Compatible with various LLM providers including DeepSeek, OpenAI, Google Gemini, and Anthropic Claude.
- **Parallel Evaluation**: Supports multi-process execution to speed up the evaluation of large datasets.
- **Advanced Agent Capabilities**:
  - **Auto-parsing**: Automatically parses table structures.
  - **Thinking Mode**: Enables chain-of-thought reasoning for complex tasks.
  - **Multi-turn Dialogue**: Supports multi-turn interactions between the user and the agent.
- **Comprehensive Metrics**: Evaluates performance based on accuracy, answer quality, tool usage, and table dependency.
- **Trace Recording**: detailed execution traces are saved for debugging and analysis.

## Project Structure

```
.
├── config/             # Configuration files
│   └── api_key.json    # API key configuration
├── dataset/            # Evaluation datasets
│   ├── tables/         # Source table files (Excel/CSV)
│   ├── samples.json    # Evaluation tasks and ground truth
│   └── README.md       # Dataset documentation
├── src/                # Source code
├── evaluate.py         # Main evaluation script
├── run_eval.sh         # Example startup script
├── config.json         # General configuration
└── README.md           # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.10+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, please install necessary packages like `pandas`, `openai`, `anthropic`, `google-generativeai`, etc., based on your usage.)*

3.  **Configure API Keys:**
    Update `config/api_key.json` with your LLM provider API keys.

4.  **Download Embedding Model:**
    You need to download the embedding model (e.g., `Qwen/Qwen3-Embedding-0.6B`) beforehand.
    By default, the system expects the model to be located at `./models/Qwen/Qwen3-Embedding-0.6B`.
    You can change this path in `config.json` under the `default_embedding_model` field.

## Configuration

The `config.json` file contains general configuration settings for the evaluation framework.

- **`tmp_root`**: Temporary directory for data storage.
- **`model_cache_dir`**: Directory to cache downloaded models.
- **`default_embedding_model`**: Path to the default embedding model (e.g., `./models/Qwen/Qwen3-Embedding-0.6B`).
- **`batch_size`**: Batch size for evaluation.
- **`model`**: Configuration for specific model names (e.g., `EMBEDDING_MODEL`, `GPT_MODEL`).
- **`chat_model`**: Default chat model configuration (provider and config key).

## Usage

### Quick Start

You can use the provided shell script to start the evaluation:

```bash
bash run_eval.sh
```

### Running `evaluate.py`

The `evaluate.py` script is the entry point for the evaluation framework. Here is an example command:

```bash
python evaluate.py \
    --mode all \
    --config_key "deepseek-v3.2" \
    --eval_file "dataset/samples.json" \
    --table_path "dataset/tables" \
    --workers 1 \
    --sample 2 \
    --max_steps 70 \
    --auto_parse True \
    --enable_thinking True \
    --verbose True \
    --trace_save_dir "traces_output"
```

### Key Arguments

- `--mode`: Execution mode. Choices:
  - `gene`: Generate traces (run the agent).
  - `eval`: Evaluate existing traces using metrics.
  - `all`: Generate traces and then evaluate them.
- `--config_key`: The configuration key for the model to be tested (defined in `config.json`).
- `--eval_file`: Path to the JSON file containing evaluation samples (default: `dataset/samples.json`).
- `--table_path`: Root directory containing the table files (default: `dataset/tables`).
- `--workers`: Number of parallel worker processes.
- `--sample`: Number of samples to run (0 for all).
- `--max_steps`: Maximum number of steps allowed for the agent per task.
- `--enable_thinking`: Enable "thinking" mode for the agent (if supported by the model).
- `--trace_save_dir`: Directory to save execution traces and results.
- `--unimodel`: Unified model config key to override all other config keys (useful for forcing a specific model).

## Dataset

Currently, we have released a portion of the tables and test samples. The full dataset will coming soon.
The dataset consists of table files and a `samples.json` file defining the tasks.
- **Tables**: Located in `dataset/tables/`. Supports `.xlsx`, `.xls`, and `.csv`.
- **Samples**: `dataset/samples.json` contains the queries, ground truth, and execution logic.

For more details on the dataset structure, please refer to [dataset/README.md](dataset/README.md).

## Output

Evaluation results and traces are saved in the directory specified by `--trace_save_dir`.
- **Traces**: JSON files containing the detailed log of the agent's actions and the user's interaction.
- **Reports**: Markdown or JSON reports summarizing the evaluation metrics (accuracy, quality, etc.).
