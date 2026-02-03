
from typing import List, Dict, Any, Union
import os
from src.evaluation.trace_analysis import calculate_tool_metrics
from src.evaluation.evaluation_judge import EvaluationJudgeLLM, SubAccJudgeLLM
from src.utils.chat_api import ChatClient
from src.utils.common import parse_json_response
from src.prompts.AgentEvalPrompt import TABLE_COVERAGE_EVAL_PROMPT

# --- Metric Registry ---
class MetricRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(metric_class):
            cls._registry[name] = metric_class
            return metric_class
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Metric {name} not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

# --- Base Metric Interface ---
class BaseMetric:
    def evaluate(self, context: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate specific aspects of the trace.
        Args:
            context: A single context dict OR a list of context dicts.
        Returns:
            A single result dict OR a list of result dicts.
        """
        raise NotImplementedError

# --- Base LLM Metric ---
class BaseLLMMetric(BaseMetric):
    def __init__(self, config_key: str, metric_name: str, step_key: str):
        self.judge = self._create_judge(config_key)
        self.metric_name = metric_name 
        self.step_key = step_key

    def _create_judge(self, config_key: str):
        raise NotImplementedError

    def _get_true_answer(self, eval_info_item: Dict[str, Any], pair: Any = None) -> str:
        """
        Subclasses can override this to extract the correct true answer field.
        Default behavior: return 'answer' field.
        """
        return eval_info_item.get('answer', '')

    def evaluate(self, context: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        is_batch = isinstance(context, list)
        contexts = context if is_batch else [context]
        
        all_queries, all_true_answers, all_model_answers = [], [], []
        valid_judge_info, missing_info = [], []
        batch_results = [{self.step_key: []} for _ in contexts]

        for ctx_idx, ctx in enumerate(contexts):
            qa_pairs = ctx.get('qa_pairs', [])
            for i, pair in enumerate(qa_pairs):
                is_missing = getattr(pair, 'is_missing', False)
                step_idx = getattr(pair, 'step_index', i + 1)
                query = getattr(pair, 'query', '')
                true_answer = getattr(pair, 'true_answer', '')
                model_answer = getattr(pair, 'answer', '')
                
                if not is_missing:
                    eval_info_list = ctx.get('eval_info') or []
                    if not ctx.get("eval_info"):
                        print(ctx)
                    # Pass pair object to allow prioritized extraction (e.g., score_points)
                    eval_info_item = eval_info_list[i] if i < len(eval_info_list) else {}
                    true_answer = self._get_true_answer(eval_info_item, pair=pair)

                if not true_answer:
                    missing_info.append((ctx_idx, step_idx, query))
                elif is_missing or (not str(model_answer).strip()):
                    missing_info.append((ctx_idx, step_idx, query))
                else:
                    all_queries.append(query)
                    all_true_answers.append(true_answer)
                    all_model_answers.append(model_answer)
                    valid_judge_info.append((ctx_idx, step_idx, query))

        judge_results = []
        if all_queries:
            judge_results = self.judge(all_queries, all_true_answers, all_model_answers)
            if not isinstance(judge_results, list):
                judge_results = [judge_results]
            
            for idx, ((ctx_idx, step_idx, query), res) in enumerate(zip(valid_judge_info, judge_results)):
                if isinstance(res, dict):
                    res['is_missing'] = False
                    res['true_answer'] = all_true_answers[idx]
                    res['model_answer'] = all_model_answers[idx]
                step_result = {
                    "step_index": step_idx, # step_idx is already 1-based from aligned_pairs
                    "query": query,
                    self.metric_name: res
                }
                batch_results[ctx_idx][self.step_key].append(step_result)
        # Generate failure template based on successful judge results
        fail_template = {"is_missing": True}
        if judge_results:
            ref_result = judge_results[0]
            if isinstance(ref_result, dict):
                for k, v in ref_result.items():
                    if isinstance(v, (int, float, bool)):
                        fail_template[k] = 0
                    else:
                        fail_template[k] = None
            fail_template["is_missing"] = True

        # Process missing entries using the template
        for (ctx_idx, step_idx, query) in missing_info:
            step_result = {
                "step_index": step_idx,
                "query": query,
                self.metric_name: fail_template.copy()
            }
            batch_results[ctx_idx][self.step_key].append(step_result)
        for res in batch_results:
            res[self.step_key].sort(key=lambda x: x['step_index'])
        return batch_results if is_batch else batch_results[0]

# --- Concrete Metrics ---
@MetricRegistry.register("tool")
class ToolMetric(BaseMetric):
    def evaluate(self, context: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(context, list):
            return [self._evaluate_single(c) for c in context]
        return self._evaluate_single(context)

    def _evaluate_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        metrics = calculate_tool_metrics(context.get('last_trace'))
        return {
            "tool_metrics": {
                "total_tokens": metrics.total_tokens,
                "tool_success_rate": metrics.tool_success_rate,
                "tool_parallelism": metrics.tool_parallelism,
                "tool_calls": metrics.tool_calls
            }
        }

@MetricRegistry.register("accuracy")
class AccuracyMetric(BaseLLMMetric):
    def __init__(self, config_key="deepseek-deepseek-v3.2"):
        super().__init__(config_key, metric_name="accuracy", step_key="accuracy_steps")

    def _create_judge(self, config_key):
        return SubAccJudgeLLM(ChatClient(config_key=config_key))
        
    def _get_true_answer(self, eval_info_item: Dict[str, Any], pair: Any = None) -> str:
        # Always use score_points as the ground truth
        if pair and getattr(pair, 'score_points', None):
            return pair.score_points
        return eval_info_item.get('score_points', '')

@MetricRegistry.register("quality")
class QualityMetric(BaseLLMMetric):
    def __init__(self, config_key="deepseek-deepseek-v3.2"):
        super().__init__(config_key, metric_name="quality", step_key="quality_steps")

    def _create_judge(self, config_key):
        return EvaluationJudgeLLM(ChatClient(config_key=config_key))

@MetricRegistry.register("table_depend")
class TableDependMetric(BaseMetric):
    def __init__(self, config_key="deepseek-deepseek-v3.2"):
        self.client = ChatClient(config_key=config_key)

    def evaluate(self, context: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        is_batch = isinstance(context, list)
        contexts = context if is_batch else [context]
        batch_results,all_prompts, all_metadata = [], [], []
        # 1. Prepare data and Prompts
        for ctx_idx, ctx in enumerate(contexts):
            qa_pairs = ctx.get('qa_pairs', [])
            for i, pair in enumerate(qa_pairs):
                step_idx = getattr(pair, 'step_index', i + 1)
                query = getattr(pair, 'query', '')
                model_tables = getattr(pair, 'data_source', []) or []
                true_tables = getattr(pair, 'related_tables', []) or []
                true_tables_basenames = [os.path.basename(t) for t in true_tables]
                model_tables_basenames = [os.path.basename(t) for t in model_tables]
                true_tables_str = "\n".join([f"- {t}" for t in true_tables_basenames]) if true_tables_basenames else "None"
                pred_tables_str = "\n".join([f"- {t}" for t in model_tables_basenames]) if model_tables_basenames else "None"
                prompt = TABLE_COVERAGE_EVAL_PROMPT.format(
                    true_tables=true_tables_str,
                    pred_tables=pred_tables_str
                )
                all_prompts.append(prompt)
                all_metadata.append({
                    "ctx_idx": ctx_idx,
                    "step_idx": step_idx,
                    "query": query,
                    "model_tables": model_tables,
                    "true_tables": true_tables,
                    "true_tables_basenames": true_tables_basenames,
                    "model_tables_basenames": model_tables_basenames
                })
            batch_results.append({"table_depend_steps": []})
        # 2. Batch call LLM
        if all_prompts:
            try:
                responses = self.client.batch_chat(all_prompts,
                                                   temperature=0.0, 
                                                   response_format={"type": "json_object"},
                                                   threads=10, batch_size=20)
            except Exception as e:
                responses = [{"content": f'{{"reasoning": "Error: {str(e)}", "covered_true_count": 0, "correct_pred_count": 0}}'} for _ in all_prompts]
        else:
            responses = []
        # 3. Process results and fill back
        for metadata, resp in zip(all_metadata, responses):
            ctx_idx = metadata["ctx_idx"]
            true_tables = metadata["true_tables"]
            true_tables_basenames = metadata["true_tables_basenames"]
            model_tables_basenames = metadata["model_tables_basenames"]
            try:
                result_json = parse_json_response(resp['content'])
                if not isinstance(result_json, dict):
                     result_json = {"reasoning": "Parse Error", "covered_true_count": 0, "correct_pred_count": 0}
            except Exception:
                result_json = {"reasoning": "Exception Error", "covered_true_count": 0, "correct_pred_count": 0}
            
            # Compatibility for old format, prioritize new fields
            covered_true_count = result_json.get("covered_true_count", result_json.get("covered_count", 0))
            correct_pred_count = result_json.get("correct_pred_count", covered_true_count) # Fallback to covered_true_count if correct_pred_count is missing (may lead to precision > 1, but as fallback)
            
            reasoning = result_json.get("reasoning", "")
            total_true = len(true_tables_basenames)
            total_pred = len(model_tables_basenames)
            
            recall = covered_true_count / total_true if total_true else 0
            precision = correct_pred_count / total_pred if total_pred else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            match_result = {
                "model_tables": metadata["model_tables"],
                "true_tables": true_tables,
                "reasoning": reasoning,
                "covered_count": covered_true_count, # Keep key for compatibility
                "total_true": total_true,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "match_result": "completed"
            }
            step_result = {
                "step_index": metadata["step_idx"],
                "query": metadata["query"],
                "table_depend": match_result
            }
            batch_results[ctx_idx]["table_depend_steps"].append(step_result)
        # 4. Sort each result to ensure correct order (due to mixed direct append and batch callbacks)
        for res in batch_results:
            res["table_depend_steps"].sort(key=lambda x: x['step_index'])
        return batch_results if is_batch else batch_results[0]
