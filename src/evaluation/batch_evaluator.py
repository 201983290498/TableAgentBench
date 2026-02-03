import os
import json
import glob
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from src.evaluation.trace_analysis import extract_metadata, parse_conversation, analyze_query_answer_pairs, get_eval_info
# --- Metric Registry & Base Metric ---
from src.evaluation.base_metric import MetricRegistry, BaseMetric

# --- Evaluator ---
class BatchEvaluator:
    def __init__(self, trace_dir, eval_file_path, config_key="xiaomi-mimo", metrics: List[Any] = None):
        self.trace_dir = trace_dir
        self.eval_file_path = eval_file_path
        self.config_key = config_key
        self.results = []
        self.metrics = []
        metrics = metrics or []
        for m in metrics:
            if isinstance(m, str):
                if m == "accuracy":
                    self.metrics.append(MetricRegistry.create(m, config_key=config_key))
                elif m == "quality":
                    self.metrics.append(MetricRegistry.create(m, config_key=config_key))
                else:
                    self.metrics.append(MetricRegistry.create(m))
            else:
                self.metrics.append(m)
        self.metrics = self.metrics if len(self.metrics) else [
            MetricRegistry.create("tool"),
            MetricRegistry.create("accuracy", config_key=config_key),
            MetricRegistry.create("quality", config_key=config_key),
            MetricRegistry.create("table_depend", config_key=config_key)
        ]

    def _load_context(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Pre-process context
            meta = extract_metadata(data)
            task_query = meta['task']
            last_trace = meta['last_trace']
            parsed_items = parse_conversation(last_trace)
            eval_info_list = get_eval_info(task_query, self.eval_file_path)
            trace_pairs = analyze_query_answer_pairs(parsed_items, eval_info_list=eval_info_list)

            context = {
                "data": data,
                "meta": meta,
                "last_trace": last_trace,
                "parsed_items": parsed_items,
                "qa_pairs": trace_pairs,
                "eval_info": eval_info_list,
                "task_query": task_query,
                "file_path": file_path
            }
            # Base result structure
            file_result = {
                "file_name": os.path.basename(file_path),
                "task": task_query,
                "success": meta['success'],
                "total_turns": meta['total_turns'],
            }
            return context, file_result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def _merge_steps(self, result):
        """Merge split metric steps into a unified eval_steps list."""
        acc_steps = result.get("accuracy_steps", [])
        qual_steps = result.get("quality_steps", [])
        dep_steps = result.get("table_depend_steps", [])
        if not acc_steps and not qual_steps and not dep_steps:
            return
        steps_map = {}
        for s in acc_steps:
            idx = s['step_index']
            if idx not in steps_map:
                steps_map[idx] = {"step_index": idx, "query": s['query']}
            steps_map[idx]["accuracy"] = s['accuracy']
            
        for s in qual_steps:
            idx = s['step_index']
            if idx not in steps_map:
                steps_map[idx] = {"step_index": idx, "query": s['query']}
            steps_map[idx]["quality"] = s['quality']
        for s in dep_steps:
            idx = s['step_index']
            if idx not in steps_map:
                steps_map[idx] = {"step_index": idx, "query": s['query']}
            steps_map[idx]["table_depend"] = s['table_depend']
        if steps_map:
            result['eval_steps'] = sorted(steps_map.values(), key=lambda x: x['step_index'])

    def evaluate_single_file(self, file_path):
        print(f"Processing: {os.path.basename(file_path)}")
        context, file_result = self._load_context(file_path)
        if not context:
            return None
        try:
            # Run all metrics
            for metric in self.metrics:
                # evaluate now supports single context too, but returns dict for single input
                file_result.update(metric.evaluate(context))
            self._merge_steps(file_result)
            return file_result
        except Exception as e:
            print(f"Error evaluating {file_path}: {e}")
            return None

    def run(self):
        json_files = glob.glob(os.path.join(self.trace_dir, "*.json"))
        print(f"Found {len(json_files)} trace files.")
        
        contexts = []
        file_results_map = {}
        # 1. Load all data
        for file_path in json_files:
            ctx, res = self._load_context(file_path)
            if ctx:
                contexts.append(ctx)
                file_results_map[file_path] = res
        if not contexts:
            return []
        for metric in self.metrics:
            try:
                # evaluate supports list input for batch processing
                batch_results = metric.evaluate(contexts)
                # Update results
                for ctx, res in zip(contexts, batch_results):
                    file_path = ctx['file_path']
                    file_results_map[file_path].update(res)
            except Exception as e:
                print(f"Error running metric {type(metric).__name__}: {e}")

        # 3. Post-process
        for file_path, res in file_results_map.items():
            self._merge_steps(res)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Add evaluation results
                data['evaluation'] = {k: v for k, v in res.items() if k != 'file_name' and k != 'eval_steps'}
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving evaluation to {file_path}: {e}")
        self.results = list(file_results_map.values())
        return self.results

    def generate_report(self, output_path):
        if not self.results:
            print("No results to generate report.")
            return
        # Calculate statistics
        total_files = len(self.results)
        success_count = sum(1 for r in self.results if r.get('success'))
        avg_turns = np.mean([r.get('total_turns', 0) or 0 for r in self.results])
        
        # Tool metrics stats
        tool_metrics_list = [r.get('tool_metrics', {}) for r in self.results]
        avg_tool_success = np.mean([m.get('tool_success_rate', 0) for m in tool_metrics_list]) if tool_metrics_list else 0
        avg_tool_parallelism = np.mean([m.get('tool_parallelism', 0) for m in tool_metrics_list]) if tool_metrics_list else 0
        
        # Token usage stats
        token_counts = [m.get('total_tokens', 0) for m in tool_metrics_list]
        avg_tokens = np.mean(token_counts) if token_counts else 0
        max_tokens = np.max(token_counts) if token_counts else 0
        min_tokens = np.min(token_counts) if token_counts else 0

        # LLM eval stats
        all_steps = []
        for r in self.results:
            all_steps.extend(r.get('eval_steps', []))
            
        def safe_float(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        if all_steps:
            # --- Micro Averages (All Steps) ---
            # Accuracy Metrics
            acc_data = [s['accuracy'] for s in all_steps if 'accuracy' in s]
            avg_strict_coverage = np.mean([safe_float(d.get('coverage_ratio', 0)) for d in acc_data]) if acc_data else 0.0
            valid_acc_data = [d for d in acc_data if not d.get('is_missing')]
            avg_coverage = np.mean([safe_float(d.get('coverage_ratio', 0)) for d in valid_acc_data]) if valid_acc_data else 0.0
            qual_data = [s['quality'] for s in all_steps if 'quality' in s]
            if qual_data:
                avg_strict_richness = np.mean([safe_float(d.get('richness_score', 0)) for d in qual_data])
                avg_strict_redundancy = np.mean([safe_float(d.get('redundancy_score', 0)) for d in qual_data])
                avg_strict_contradiction = np.mean([safe_float(d.get('contradiction_score', 0)) for d in qual_data])
            else:
                avg_strict_richness = avg_strict_redundancy = avg_strict_contradiction = 0.0
            valid_qual_data = [d for d in qual_data if not d.get('is_missing')]
            if valid_qual_data:
                avg_richness = np.mean([safe_float(d.get('richness_score', 0)) for d in valid_qual_data])
                avg_redundancy = np.mean([safe_float(d.get('redundancy_score', 0)) for d in valid_qual_data])
                avg_contradiction = np.mean([safe_float(d.get('contradiction_score', 0)) for d in valid_qual_data])
            else:
                avg_richness = avg_redundancy = avg_contradiction = 0.0
        else:
            avg_coverage = avg_strict_coverage = avg_richness = avg_redundancy = avg_contradiction = 0
            avg_strict_richness = avg_strict_redundancy = avg_strict_contradiction = 0

        # --- Macro Averages (Per Trace) ---
        trace_means = {
            'coverage': [], 'strict_coverage': [],
            'richness': [], 'redundancy': [], 'contradiction': [],
            'strict_richness': [], 'strict_redundancy': [], 'strict_contradiction': []
        }

        for r in self.results:
            steps = r.get('eval_steps', [])
            if not steps:
                continue
            
            # Get metric values for this trace
            acc_metrics = [s['accuracy'] for s in steps if 'accuracy' in s]
            if acc_metrics:
                # Strict
                trace_means['strict_coverage'].append(np.mean([safe_float(m.get('coverage_ratio', 0)) for m in acc_metrics]))
                # Ordinary
                valid_acc = [m for m in acc_metrics if not m.get('is_missing')]
                if valid_acc:
                    trace_means['coverage'].append(np.mean([safe_float(m.get('coverage_ratio', 0)) for m in valid_acc]))
            
            qual_metrics = [s['quality'] for s in steps if 'quality' in s]
            if qual_metrics:
                # Strict
                trace_means['strict_richness'].append(np.mean([safe_float(m.get('richness_score', 0)) for m in qual_metrics]))
                trace_means['strict_redundancy'].append(np.mean([safe_float(m.get('redundancy_score', 0)) for m in qual_metrics]))
                trace_means['strict_contradiction'].append(np.mean([safe_float(m.get('contradiction_score', 0)) for m in qual_metrics]))
                
                # Ordinary
                valid_qual = [m for m in qual_metrics if not m.get('is_missing')]
                if valid_qual:
                    trace_means['richness'].append(np.mean([safe_float(m.get('richness_score', 0)) for m in valid_qual]))
                    trace_means['redundancy'].append(np.mean([safe_float(m.get('redundancy_score', 0)) for m in valid_qual]))
                    trace_means['contradiction'].append(np.mean([safe_float(m.get('contradiction_score', 0)) for m in valid_qual]))

        # Compute Macro Averages
        macro_coverage = np.mean(trace_means['coverage']) if trace_means['coverage'] else 0
        macro_strict_coverage = np.mean(trace_means['strict_coverage']) if trace_means['strict_coverage'] else 0
        
        macro_richness = np.mean(trace_means['richness']) if trace_means['richness'] else 0
        macro_redundancy = np.mean(trace_means['redundancy']) if trace_means['redundancy'] else 0
        macro_contradiction = np.mean(trace_means['contradiction']) if trace_means['contradiction'] else 0
        
        macro_strict_richness = np.mean(trace_means['strict_richness']) if trace_means['strict_richness'] else 0
        macro_strict_redundancy = np.mean(trace_means['strict_redundancy']) if trace_means['strict_redundancy'] else 0
        macro_strict_contradiction = np.mean(trace_means['strict_contradiction']) if trace_means['strict_contradiction'] else 0

        # Count strict/non-strict samples
        count_strict = len(acc_data) if 'acc_data' in locals() else 0
        count_non_strict = len(valid_acc_data) if 'valid_acc_data' in locals() else 0

        # Table Depend Metrics
        dep_data = []
        valid_dep_data = []
        
        for s in all_steps:
            if 'table_depend' in s:
                dep_res = s['table_depend']
                # Determine if step is missing based on accuracy metric
                is_missing = False
                if 'accuracy' in s:
                    is_missing = s['accuracy'].get('is_missing', False)
                
                dep_data.append(dep_res)
                if not is_missing:
                    valid_dep_data.append(dep_res)

        # Strict (All steps)
        if dep_data:
            avg_strict_table_recall = np.mean([safe_float(d.get('recall', 0)) for d in dep_data])
            avg_strict_table_precision = np.mean([safe_float(d.get('precision', 0)) for d in dep_data])
            avg_strict_table_f1 = np.mean([safe_float(d.get('f1', 0)) for d in dep_data])
        else:
            avg_strict_table_recall = avg_strict_table_precision = avg_strict_table_f1 = 0.0

        # Non-Strict (Ignore Missing)
        if valid_dep_data:
            avg_table_recall = np.mean([safe_float(d.get('recall', 0)) for d in valid_dep_data])
            avg_table_precision = np.mean([safe_float(d.get('precision', 0)) for d in valid_dep_data])
            avg_table_f1 = np.mean([safe_float(d.get('f1', 0)) for d in valid_dep_data])
        else:
            avg_table_recall = avg_table_precision = avg_table_f1 = 0.0

        # Find High Accuracy but Low Table Recall cases
        high_acc_low_recall_cases = []
        for r in self.results:
            file_name = r.get('file_name', 'unknown')
            steps = r.get('eval_steps', [])
            for s in steps:
                acc_res = s.get('accuracy', {})
                dep_res = s.get('table_depend', {})
                
                cov = safe_float(acc_res.get('coverage_ratio', 0))
                recall = safe_float(dep_res.get('recall', 0))
                
                # Check condition: Coverage == 1.0 AND Recall < 1.0
                # We also check if table_depend data actually exists to avoid false positives on steps without table eval
                if 'accuracy' in s and 'table_depend' in s and cov >= 1.0 and recall < 1.0:
                    true_tables = dep_res.get('true_tables', [])
                    model_tables = dep_res.get('model_tables', [])
                    true_tables_basenames = [os.path.basename(t) for t in true_tables]
                    model_tables_basenames = [os.path.basename(t) for t in model_tables]
                    
                    high_acc_low_recall_cases.append({
                        "file_name": file_name,
                        "step_index": s.get('step_index'),
                        "query": s.get('query'),
                        "coverage": cov,
                        "recall": recall,
                        "reasoning": acc_res.get('reasoning', ''),
                        "true_answer": acc_res.get('true_answer', ''),
                        "model_answer": acc_res.get('model_answer', ''),
                        "true_tables": true_tables_basenames,
                        "model_tables": model_tables_basenames
                    })

        # Calculate Task Pass@1 and Turn Coverage@1
        task_pass_at_1_count = 0
        total_valid_tasks = 0
        all_coverage_ratios = []
        for r in self.results:
            steps = r.get('eval_steps', [])
            if not steps:
                continue
            total_valid_tasks += 1
            task_passed = True
            for s in steps:
                if 'accuracy' in s:
                    cov = safe_float(s['accuracy'].get('coverage_ratio', 0))
                    all_coverage_ratios.append(cov)
                    if cov < 1.0:
                        task_passed = False
                else:
                    task_passed = False
            if task_passed:
                task_pass_at_1_count += 1
        task_pass_at_1_rate = task_pass_at_1_count / total_valid_tasks if total_valid_tasks else 0.0
        turn_coverage_at_1_count = sum(1 for c in all_coverage_ratios if c >= 1.0)
        turn_coverage_at_1_rate = turn_coverage_at_1_count / len(all_coverage_ratios) if all_coverage_ratios else 0.0

        # Generate Markdown
        md_content = f"""# Table Agent Batch Evaluation Report

## 1. Overview
- **Evaluation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Cases**: {total_files}
- **Success Rate**: {success_count/total_files:.2%} ({success_count}/{total_files})
- **Avg Turns**: {avg_turns:.2f}
- **Task Pass@1**: {task_pass_at_1_rate:.2%} ({task_pass_at_1_count}/{total_valid_tasks})
- **Turn Coverage@1**: {turn_coverage_at_1_rate:.2%}

## 2. Tool Metrics
- **Avg Tool Success Rate**: {avg_tool_success:.2%}
- **Avg Tool Parallelism**: {avg_tool_parallelism:.2f}
- **Token Usage**:
  - Avg: {avg_tokens:.2f}
  - Max: {max_tokens}
  - Min: {min_tokens}

## 3. Quality Metrics (LLM Judge)
Based on {len(all_steps)} evaluated steps across {total_files} traces:
- **Strict Count (All Steps)**: {count_strict}
- **Non-Strict Count (Ignore Missing)**: {count_non_strict}

| Metric | Micro-Avg (All Steps) | Macro-Avg (Per Trace) | Description |
|--------|-----------------------|-----------------------|-------------|
| **Coverage** | {avg_coverage:.3f} | {macro_coverage:.3f} | Ratio of covered metrics (Ignore Missing) |
| **Strict Coverage** | {avg_strict_coverage:.3f} | {macro_strict_coverage:.3f} | Ratio of covered metrics (Include Missing) |
| **Richness** | {avg_richness:.2f}/5 | {macro_richness:.2f}/5 | Completeness and reasoning |
| **Strict Richness** | {avg_strict_richness:.2f}/5 | {macro_strict_richness:.2f}/5 | Completeness (missing=0) |
| **Redundancy** | {avg_redundancy:.2f}/5 | {macro_redundancy:.2f}/5 | Conciseness (higher is better) |
| **Strict Redundancy** | {avg_strict_redundancy:.2f}/5 | {macro_strict_redundancy:.2f}/5 | Conciseness (missing=0) |
| **Contradiction** | {avg_contradiction:.2f}/5 | {macro_contradiction:.2f}/5 | Consistency (higher is better) |
| **Strict Contradiction** | {avg_strict_contradiction:.2f}/5 | {macro_strict_contradiction:.2f}/5 | Consistency (missing=0) |
| **Table Recall** | {avg_table_recall:.3f} | - | Table Dependency Recall (Ignore Missing) |
| **Strict Table Recall** | {avg_strict_table_recall:.3f} | - | Table Dependency Recall (Include Missing) |
| **Table Precision** | {avg_table_precision:.3f} | - | Table Dependency Precision (Ignore Missing) |
| **Strict Table Precision** | {avg_strict_table_precision:.3f} | - | Table Dependency Precision (Include Missing) |
| **Table F1** | {avg_table_f1:.3f} | - | Table Dependency F1 Score (Ignore Missing) |
| **Strict Table F1** | {avg_strict_table_f1:.3f} | - | Table Dependency F1 Score (Include Missing) |

## 4. Special Cases (High Accuracy, Low Table Recall)
Steps with Coverage=1.0 but Table Recall < 1.0:
"""
        if high_acc_low_recall_cases:
            for case in high_acc_low_recall_cases:
                md_content += f"- **{case['file_name']}** (Step {case['step_index']})\n"
                md_content += f"  - Query: {case['query']}\n"
                md_content += f"  - Coverage: {case['coverage']} | Recall: {case['recall']:.3f}\n"
                md_content += f"  - True Answer: {case['true_answer']}\n"
                md_content += f"  - Model Answer: {case['model_answer']}\n"
                md_content += f"  - True Tables: {case['true_tables']}\n"
                md_content += f"  - Predicted Tables: {case['model_tables']}\n"
                md_content += f"  - Reasoning: {case['reasoning'][:200]}...\n"
        else:
            md_content += "No such cases found.\n"

        md_content += "\n## 5. Detailed Cases\n"

        for r in self.results:
            md_content += f"### {r['file_name']}\n"
            md_content += f"- **Task**: {r.get('task', '')[:100]}...\n"
            md_content += f"- **Success**: {'✅' if r.get('success') else '❌'}\n"
            tool_m = r.get('tool_metrics', {})
            md_content += f"- **Tool Success**: {tool_m.get('tool_success_rate', 0):.2%}\n"
            eval_steps = r.get('eval_steps', [])
            if eval_steps:
                md_content += "\n  **Step Evaluation**:\n"
                for step in eval_steps:
                    md_content += f"  - **Step {step['step_index']}**:\n"
                    if 'accuracy' in step:
                        is_miss = step['accuracy'].get('is_missing', False)
                        status = "MISSING" if is_miss else "PRESENT"
                        md_content += f"    - Coverage: {step['accuracy'].get('coverage_ratio')} ({step['accuracy'].get('covered_metrics')}/{step['accuracy'].get('total_metrics')}) [{status}]\n"
                        if step['accuracy'].get('reasoning'):
                            md_content += f"    - *Reasoning*: {step['accuracy'].get('reasoning')}\n"
                    
                    if 'quality' in step:
                        is_miss = step['quality'].get('is_missing', False)
                        status = "MISSING" if is_miss else "PRESENT"
                        md_content += f"    - Quality: R={step['quality'].get('richness_score')} / Red={step['quality'].get('redundancy_score')} / C={step['quality'].get('contradiction_score')} [{status}]\n"
            else:
                md_content += "\n  *(No detailed evaluation data)*\n"
            md_content += "\n---\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Report generated at: {output_path}")