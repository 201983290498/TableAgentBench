import os
import sys
import json
import argparse
import time
import multiprocessing
import traceback
import tempfile
# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.evaluation.base_metric import MetricRegistry
from src.agents.table_agent import create_table_agent
from src.agents.user_agent import UserAgent, UserAgentConfig
from src.agents.orchestrator import MultiTurnOrchestrator
from src.utils.chat_api import ChatClient
from src.utils.gemini_client import GeminiClient
from src.utils.claude_client import ClaudeClient
from src.evaluation.batch_evaluator import BatchEvaluator
from src.retrival.embedder_service import EmbeddingServiceManager, set_shared_queues
from src.retrival.embedder import enable_remote_client_mode
from src.utils.common import read_config

# Define temporary root directory path
config = read_config()
TMP_ROOT = config.get("tmp_root", "/tmp/data")
os.makedirs(TMP_ROOT, exist_ok=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def worker_initializer(req_queue, resp_queue):
    """
    Subprocess initializer
    Called when each worker starts to set up shared queues and enable remote Client mode
    """
    set_shared_queues(req_queue, resp_queue)
    enable_remote_client_mode()
    print(f"[Worker {os.getpid()}] Initialized with shared embedding service")

def link_files_recursively(src_dir, dst_dir):
    """
    Recursively link all files from src_dir to dst_dir
    """
    if not os.path.exists(src_dir):
        print(f"[Warning] Source path does not exist: {src_dir}")
        return

    abs_src = os.path.abspath(src_dir)
    for root, dirs, files in os.walk(abs_src):
        rel_root = os.path.relpath(root, abs_src)
        target_root = os.path.join(dst_dir, rel_root)
        
        if rel_root != ".":
            os.makedirs(target_root, exist_ok=True)
            
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_root, file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            os.symlink(src_file, dst_file)

def run_single_eval(args_tuple):
    """
    Execution function for a single evaluation task (runs in subprocess)
    """
    sample, args, task_id = args_tuple
    
    # 1. Prepare working directory
    original_table_path = sample['file_path']
    
    # Apply unimodel config override in subprocess, as main process monkey patch is not inherited in spawn mode
    if args.unimodel:
        modify_config_key(args)
    
    try:
        # Use TemporaryDirectory to automatically manage creation and cleanup of temporary directories
        with tempfile.TemporaryDirectory(dir=TMP_ROOT, prefix=f"task_") as task_work_dir:
            target_path = os.path.join(task_work_dir, original_table_path)
            os.makedirs(target_path, exist_ok=True)
            link_files_recursively(original_table_path, target_path)
            sample['file_path'] = target_path
            
            user_config = UserAgentConfig(
                model_provider=args.user_provider,
                model_name=args.user_model,
                config_key=args.user_model
            )
            user_agent = UserAgent(user_config)
            llm_client = None
            if "gemini" in args.config_key:
                llm_client = GeminiClient(config_key=args.config_key)
            elif "claude" in args.config_key:
                llm_client = ClaudeClient(config_key=args.config_key)
            else:
                llm_client = ChatClient(config_key=args.config_key)
            table_agent = create_table_agent(
                llm_client=llm_client or ChatClient(config_key=args.config_key),
                enable_thinking=args.enable_thinking,
                auto_parse=args.auto_parse,
                max_steps=args.max_steps,
                verbose=args.verbose,
                auto_generate_planning=False,
                trace_save_dir=args.trace_save_dir,
                multi_turn_mode=True,
                reset_env=True,
                max_history_tokens=args.max_history_tokens,
                include_tools=args.include_tools.split(",") if args.include_tools else None,
            )
            
            orchestrator = MultiTurnOrchestrator(user_agent, table_agent)
            
            # 5. Run evaluation
            print(f"[Task {task_id}] Start running sample: {sample.get('id')}")
            result = orchestrator.run_eval(sample, args.trace_save_dir)
            print(f"[Task {task_id}] Finish sample: {sample.get('id')}")
            return result
        
    except Exception as e:
        print(f"[Task {task_id}] Error: {e}")
        traceback.print_exc()
        return None

def load_samples(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load samples: {e}")
        return []

def get_exists_traces(trace_dir):
    """Check if traces already exist"""
    if not os.path.exists(trace_dir):
        return []
    exist_queries = set()
    for filename in os.listdir(trace_dir):
        if filename.startswith("trace_") and filename.endswith(".json"):
            try:
                data = json.load(open(os.path.join(trace_dir, filename), "r", encoding="utf-8"))
                query = data.get("metadata", {}).get("query")
                if query:
                    exist_queries.add(query.strip())
            except Exception:
                continue
    print(f"[Info] Found {len(exist_queries)} existing task records in {trace_dir}.")
    return list(exist_queries)

def modify_config_key(args):
    print(f"Enable unimodel mode, unified config_key: {args.unimodel}")
    import src.utils.common
    original_read_config = src.utils.common.read_config
    def patched_read_config(config_path: str = None):
        config = original_read_config(config_path)
        if "chat_model" in config:
            config["chat_model"]["config_key"] = args.unimodel
        return config
    src.utils.common.read_config = patched_read_config
    import src.utils.chat_api
    # Fix: Since chat_api uses from ... import read_config, must also modify reference in chat_api
    src.utils.chat_api.read_config = patched_read_config
    
    if src.utils.chat_api._chat_client_instance is not None:
        print("Resetting ChatClient singleton...")
        src.utils.chat_api._chat_client_instance = None
         
def main():
    parser = argparse.ArgumentParser(description="TableAgent Parallel Multi-turn Evaluation")
    parser.add_argument("--mode",  choices=["gene", "eval", "all"], default="gene", help="Run mode")
    parser.add_argument("--config_key", default="deepseek-v3.2", help="LLM config key")
    parser.add_argument("--user_model", type=str, default="deepseek-v3.2", help="UserAgent model")
    parser.add_argument("--user_provider", type=str, default="openai", help="UserAgent provider")
    parser.add_argument("--eval_file", type=str, default=r"dataset/T2R_multi/T2R_V2.json", help="Input sample path")
    parser.add_argument("--trace_save_dir", type=str,  default=r"dataset/T2R_multi/trace", help="Result save directory")
    parser.add_argument("--eval_config_key", default=None, help="Evaluation config key, used to select different evaluation models")
    parser.add_argument("--table_path", default="dataset/T2R", help="Original table data directory")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps")
    parser.add_argument("--auto_parse", type=str2bool, default=True, help="Whether to auto parse")
    parser.add_argument('--enable_thinking', type=str2bool, default=False, help="Whether to enable thinking mode")
    parser.add_argument('--max_history_tokens', type=int, default=127000, help="Max history tokens")
    parser.add_argument("--sample", type=int, default=0, help="Limit running sample count, 0 for unlimited")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to show verbose logs")
    parser.add_argument("--workers", type=int, default=5, help="Parallel worker count")
    parser.add_argument("--include_tools", type=str, default=None, help="Comma-separated tool name list")
    parser.add_argument('--remove_duplicates', type=str2bool, default=True, help="Whether to enable evaluation deduplication")
    parser.add_argument("--unimodel", type=str, default="deepseek-v3.2", help="Unified model config key, overrides all config_key if not empty")
    args = parser.parse_args()
    if args.table_path:
        # Use relative path (relative to current working directory) instead of absolute path
        args.table_path = os.path.relpath(os.path.abspath(args.table_path), os.getcwd())
    # Handle unimodel parameter override logic
    if args.unimodel:
        modify_config_key(args)
    print("Run args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    os.makedirs(args.trace_save_dir, exist_ok=True)
    # Variable to store shared queues
    req_queue, resp_queue = None, None
    
    if args.mode in ['gene', 'all']:
        # Start Embedding Service (Main Process)
        from src.utils.common import get_default_embedding_model
        model_name = get_default_embedding_model() or "tfidf"
        print(f"Start Embedding Service: {model_name}")
        req_queue, resp_queue = EmbeddingServiceManager.start_service(model_name=model_name, device=None)
        # Load samples
        all_samples = load_samples(args.eval_file)
        print(f"Total loaded {len(all_samples)} samples")

        # Internal deduplication: Prevent duplicate tasks in source data
        unique_samples, seen_tasks = [], set()
        for s in all_samples:
            t = s.get("task")
            if t not in seen_tasks:
                seen_tasks.add(t)
                unique_samples.append(s)
            else:
                print(f"[Warning] Found duplicate task in source data, skipped: {t[:50]}...")
        if len(unique_samples) < len(all_samples):
             print(f"Remaining {len(unique_samples)} samples after source data deduplication")
        all_samples = unique_samples

        if args.sample > 0:
            all_samples = all_samples[:args.sample]
            print(f"Limit running first {args.sample} samples")

        if args.remove_duplicates:
            exist_traces = get_exists_traces(args.trace_save_dir)
            all_samples = [s for s in all_samples if s.get("task") not in exist_traces]
            print(f"Remaining {len(all_samples)} samples after deduplication")
        tasks = []
        for i, sample in enumerate(all_samples):
            if "id" not in sample:
                sample["id"] = f"sample_{i}"
                sample['file_path'] = args.table_path if args.table_path else sample['file_path']
            # Use i as task_id
            tasks.append((sample, args, i))
        print(f"Ready to start parallel evaluation, concurrency: {args.workers}")
        print(f"Temporary working directory root: {TMP_ROOT}")
        print(f"[Main Process] Using shared Embedding Service, all subprocesses share the same GPU model")
        start_time = time.time()
        # Use multiprocessing pool, pass shared queues to subprocesses via initializer
        try:
            with multiprocessing.Pool(processes=args.workers, initializer=worker_initializer, initargs=(req_queue, resp_queue)) as pool:
                results = pool.map(run_single_eval, tasks)
        finally:
            EmbeddingServiceManager.stop_service()
        end_time = time.time()
        print(f"All tasks completed. Total time: {end_time - start_time:.2f} seconds")
    # 6. Run batch evaluation
    if args.mode in ['eval', 'all']:
        print(f"\n{'='*50}\nRunning batch evaluation...\n" + f"{'='*50}")
        evaluator = BatchEvaluator(
            trace_dir=args.trace_save_dir, 
            eval_file_path=args.eval_file, 
            config_key=args.eval_config_key or args.config_key,
            metrics=[
                MetricRegistry.create("tool"),
                MetricRegistry.create("accuracy", config_key=args.eval_config_key or args.config_key),
                MetricRegistry.create("quality", config_key=args.eval_config_key or args.config_key),
                MetricRegistry.create("table_depend", config_key=args.eval_config_key or args.config_key)
            ]
        )
        evaluator.run()
        report_path = os.path.join(args.trace_save_dir, "batch_eval_report.md")
        evaluator.generate_report(report_path)
        print(f"Evaluation completed, report generated at: {report_path}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
