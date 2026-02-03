# TableAgent 评估框架

[English](README.md) | 中文

本仓库包含一个用于评估基于 LLM 的 Agent 在表格相关任务（Table QA）上的表现的框架。旨在评估 Agent 与结构化数据（Excel, CSV）交互以回答复杂自然语言查询的能力。

## 特性

- **多模型支持**：兼容多种 LLM 提供商，包括 DeepSeek、OpenAI、Google Gemini 和 Anthropic Claude。
- **并行评估**：支持多进程执行，加速大数据集的评估。
- **高级 Agent 能力**：
  - **自动解析**：自动解析表格结构。
  - **思考模式**：支持复杂任务的思维链（Chain-of-Thought）推理。
  - **多轮对话**：支持用户与 Agent 之间的多轮交互。
- **综合指标**：基于准确性、回答质量、工具使用和表格依赖性评估性能。
- **Trace 记录**：保存详细的执行 Trace 用于调试和分析。

## 项目结构

```
.
├── config/             # 配置文件
│   └── api_key.json    # API key 配置
├── dataset/            # 评估数据集
│   ├── tables/         # 源表格文件 (Excel/CSV)
│   ├── samples.json    # 评估任务和标准答案
│   └── README.md       # 数据集文档
├── src/                # 源代码
├── evaluate.py         # 主评估脚本
├── run_eval.sh         # 启动脚本示例
├── config.json         # 通用配置
└── README.md           # 本文件
```

## 安装

1.  **克隆仓库：**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **安装依赖：**
    确保你安装了 Python 3.10+。
    ```bash
    pip install -r requirements.txt
    ```
    *（注意：如果缺少 `requirements.txt`，请根据你的使用情况手动安装必要的包，如 `pandas`, `openai`, `anthropic`, `google-generativeai` 等。）*

3.  **配置 API Key：**
    在 `config/api_key.json` 中更新你的 LLM 提供商 API Key。

4.  **下载 Embedding 模型：**
    你需要提前下载 Embedding 模型（例如 `Qwen/Qwen3-Embedding-0.6B`）。
    默认情况下，系统期望模型位于 `./models/Qwen/Qwen3-Embedding-0.6B`。
    你可以在 `config.json` 的 `default_embedding_model` 字段中更改此路径。

## 配置说明

`config.json` 文件包含评估框架的通用配置设置。

- **`tmp_root`**：临时数据存储目录。
- **`model_cache_dir`**：模型缓存目录。
- **`default_embedding_model`**：默认 Embedding 模型路径（例如 `./models/Qwen/Qwen3-Embedding-0.6B`）。
- **`batch_size`**：评估批次大小。
- **`model`**：特定模型名称的配置（例如 `EMBEDDING_MODEL`, `GPT_MODEL`）。
- **`chat_model`**：默认聊天模型配置（提供商和配置 Key）。

## 使用方法

### 快速开始

你可以使用提供的 Shell 脚本启动评估：

```bash
bash run_eval.sh
```

### 运行 `evaluate.py`

`evaluate.py` 脚本是评估框架的入口点。以下是一个命令示例：

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

### 关键参数

- `--mode`：执行模式。选项：
  - `gene`：生成 Trace（运行 Agent）。
  - `eval`：使用指标评估现有 Trace。
  - `all`：先生成 Trace，然后进行评估。
- `--config_key`：要测试的模型的配置 Key（在 `config.json` 中定义）。
- `--eval_file`：包含评估样本的 JSON 文件路径（默认：`dataset/samples.json`）。
- `--table_path`：包含表格文件的根目录（默认：`dataset/tables`）。
- `--workers`：并行工作进程的数量。
- `--sample`：运行的样本数量（0 表示全部）。
- `--max_steps`：每个任务允许 Agent 执行的最大步骤数。
- `--enable_thinking`：启用 Agent 的“思考”模式（如果模型支持）。
- `--trace_save_dir`：保存执行 Trace 和结果的目录。
- `--unimodel`：统一模型配置 Key，用于覆盖所有其他配置 Key（用于强制指定特定模型）。

## 数据集

目前，我们开放了部分的表格和测试样例。全量数据将在持续开放。
数据集由表格文件和定义任务的 `samples.json` 文件组成。
- **Tables**：位于 `dataset/tables/`。支持 `.xlsx`, `.xls`, 和 `.csv`。
- **Samples**：`dataset/samples.json` 包含查询、标准答案和执行逻辑。

有关数据集结构的更多详细信息，请参阅 [dataset/README.md](dataset/README.md)。

## 输出

评估结果和 Trace 将保存在 `--trace_save_dir` 指定的目录中。
- **Traces**：包含 Agent 动作和用户交互详细日志的 JSON 文件。
- **Reports**：Markdown 或 JSON 报告，总结评估指标（准确性、质量等）。
