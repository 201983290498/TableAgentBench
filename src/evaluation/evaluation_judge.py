import json
import sys
import os
from typing import Union, List

# Add project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.function_llm.base_llm import BaseLLM
from src.utils.chat_api import ChatClient
from src.utils.common import parse_json_response
from src.prompts.AgentEvalPrompt import ANSWER_LLM_JUDGE, SCORE_POINTS_ACC_JUDGE

class SubAccJudgeLLM(BaseLLM):
    """
    LLM-based sub-problem correctness/coverage evaluator
    """
    
    def __init__(self, client: ChatClient = None):
        """
        Initialize the evaluator
        
        Args:
            client: ChatClient instance
        """
        super().__init__(client=client, system_prompt="You are a rigorous AI evaluation expert.")
        self.prompt_template = SCORE_POINTS_ACC_JUDGE

    def __call__(self, query: Union[str, list[str]], true_answer: Union[str, list[str]], model_answer: Union[str, list[str]]) -> Union[dict, list[dict]]:
        """
        Evaluate model answers (supports single or batch processing)
        
        Args:
            query: User question or list of questions
            true_answer: True/reference answer or list of answers
            model_answer: Model-generated answer or list of answers
            
        Returns:
            Union[dict, list[dict]]: Dictionary or list of dictionaries containing scores and reasoning
        """
        is_batch = isinstance(query, list)
        # Unified list conversion
        queries = query if is_batch else [query]
        true_answers = true_answer if is_batch else [true_answer]
        model_answers = model_answer if is_batch else [model_answer]
        if not (len(queries) == len(true_answers) == len(model_answers)):
            raise ValueError("Batch input lists must have the same length")
        
        # Construct prompts
        prompts = [
            self.prompt_template.format(
                query=q,
                answer=a,
                model_answer=ma
            ) for q, a, ma in zip(queries, true_answers, model_answers)
        ]
        # Batch call LLM
        responses = self.client.batch_chat(
            prompts=prompts,
            system=self.system_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
            threads=10, batch_size=20
        )
        
        results = []
        for response in responses:
            try:
                content = response.get("content", "")
                result = parse_json_response(content)
                if "error" in result and result.get("error") == "Invalid JSON format":
                     raise ValueError("Invalid JSON format")
                results.append(result)
            except Exception as e:
                print(f"Error parsing judge response: {e}")
                print(f"Raw response: {response.get('content')}")
                results.append({
                    "total_metrics": 0,
                    "covered_metrics": 0,
                    "coverage_ratio": 0.0,
                    "reasoning": f"Evaluation failed: {str(e)}"
                })
        return results if is_batch else results[0]

class EvaluationJudgeLLM(BaseLLM):
    """
    LLM-based answer evaluator
    """
    
    def __init__(self, client: ChatClient = None):
        super().__init__(client=client, system_prompt="You are a rigorous AI evaluation expert.")
        self.prompt_template = ANSWER_LLM_JUDGE

    def __call__(self, query: Union[str, list[str]], true_answer: Union[str, list[str]], model_answer: Union[str, list[str]]) -> Union[dict, list[dict]]:
        """
        Evaluate model answers (supports single or batch processing)
        
        Args:
            query: User question or list of questions
            true_answer: True/reference answer or list of answers
            model_answer: Model-generated answer or list of answers
            
        Returns:
            Union[dict, list[dict]]: Dictionary or list of dictionaries containing scores and reasoning
        """
        is_batch = isinstance(query, list)
        
        # Unified list conversion
        queries = query if is_batch else [query]
        true_answers = true_answer if is_batch else [true_answer]
        model_answers = model_answer if is_batch else [model_answer]
        if not (len(queries) == len(true_answers) == len(model_answers)):
            raise ValueError("Batch input lists must have the same length")
        # Construct prompts
        prompts = [
            self.prompt_template.format(query=q,answer=a,model_answer=ma) for q, a, ma in zip(queries, true_answers, model_answers)
        ]
        # Batch call LLM
        responses = self.client.batch_chat(
            prompts=prompts,
            system=self.system_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
            threads=10, batch_size=20
        )
        results = []
        for response in responses:
            try:
                content = response.get("content", "")
                result = parse_json_response(content)
                if "error" in result and result.get("error") == "Invalid JSON format":
                     raise ValueError("Invalid JSON format")
                results.append(result)
            except Exception as e:
                print(f"Error parsing judge response: {e}\nRaw response: {response.get('content')}")
                results.append({
                    "richness_score": 0,
                    "richness_reason": "Parsing failed",
                    "redundancy_score": 0,
                    "redundancy_reason": "Parsing failed",
                    "contradiction_score": 0,
                    "contradiction_reason": "Parsing failed",
                    "overall_comment": f"Evaluation failed: {str(e)}"
                })
        return results if is_batch else results[0]
