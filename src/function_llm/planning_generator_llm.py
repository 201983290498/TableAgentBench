from typing import List, Dict, Any, Union, Optional
from src.function_llm.base_llm import BaseLLM
from src.utils.table_process import format_table_desc
from src.utils.common import parse_json_response
from src.prompts.ToolTemplatePrompt import PLANNING_GENERATION_PROMPT
from src.utils.chat_api import ChatClient


class PlanningGeneratorLLM(BaseLLM):
    """
    LLM class for dynamically generating execution plans based on user questions and table information.
    
    Automatically generates appropriate analysis steps based on question complexity:
    - Simple questions: 2-3 steps
    - Medium questions: 3-5 steps  
    - Complex questions: 5-7 steps
    """
    
    def __init__(
        self, 
        provider: Optional[str] = None, 
        config_key: str = "xiaomi-mimov4", 
        system_prompt: Optional[str] = None
    ):
        if system_prompt is None:
            system_prompt = PLANNING_GENERATION_PROMPT
        super().__init__(ChatClient(provider, config_key), system_prompt=system_prompt)
    
    def _format_tools_info(self, tools: Union[str, List]) -> str:
        if isinstance(tools, str):
            return tools
        return "\n".join([
            f"- {t['name']}: {t.get('description', '')}" if isinstance(t, dict) else str(t)
            for t in tools
        ])
    
    def __call__(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        threads: int = 4,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates execution plans; supports single or batch inputs.
        
        Args:
            data: Data item containing: question, table_info (optional), and available_tools (optional)
            
        Returns:
            Result contains: analysis, complexity, and steps (each containing step_id and description)
        """
        is_single = isinstance(data, dict)
        data_list = [data] if is_single else data
        
        prompts = []
        for item in data_list:
            prompt = self.system_prompt.format(
                question=item.get('question', ''),
                table_info=format_table_desc(item.get('table_info', 'Table information not provided')),
                available_tools=self._format_tools_info(item.get('available_tools', 'No specific tool restrictions'))
            )
            prompts.append(prompt)
        
        responses = self.client.batch_chat(prompts=prompts, threads=threads, verbose=not is_single, **kwargs)
        # Extract content field for parsing
        results = [parse_json_response(resp_dict["content"]) for resp_dict in responses]
        return results[0] if is_single else results
    
    def generate_steps_only(self, question: str, table_info: Union[str, Dict, List] = "", **kwargs) -> List[str]:
        result = self({"question": question, "table_info": table_info}, **kwargs)
        return [f"{s.get('step_id', i+1)}. {s.get('description', '')}" for i, s in enumerate(result.get("steps", []))]
