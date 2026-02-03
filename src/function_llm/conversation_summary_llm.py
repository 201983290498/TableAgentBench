from typing import List, Dict, Any, Union, Optional
from src.function_llm.base_llm import BaseLLM
from src.utils.common import format_conversation_history
from src.prompts.ToolTemplatePrompt import CONVERSATION_SUMMARY_PROMPT
from src.utils.chat_api import ChatClient


class ConversationSummaryLLM(BaseLLM):
    """
    LLM class for compressing conversation history
    Extracts key information from the conversation history based on the original question and planning information to generate a compressed summary.
    """
    
    def __init__(self, provider: Optional[str] = None, config_key: str = None, system_prompt: str = None):
        """
        Initialize the conversation summary LLM
        
        Args:
            provider: LLM provider
            config_key: Configuration key
            system_prompt: System prompt, uses default if None
        """
        if system_prompt is None:
            system_prompt = CONVERSATION_SUMMARY_PROMPT
        super().__init__(ChatClient(provider, config_key), system_prompt=system_prompt)
    
    def __call__(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        threads: int = 4,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Compresses conversation history and returns summary text; supports single or batch inputs.
        
        Args:
            data: Single data item (Dict) or list of data items (List[Dict])
                  Each item should contain: question, conversation_history, and planning (optional)
            threads: Number of concurrent threads for batch processing
            
        Returns:
            Single result (str) for single input, list of results (List[str]) for list input
        """
        is_single = isinstance(data, dict)
        data_list = [data] if is_single else data
        
        def format_history(conversation_history) -> str:
            if isinstance(conversation_history, str):
                return conversation_history
            elif isinstance(conversation_history, list):
                history_dicts = []
                for msg in conversation_history:
                    if isinstance(msg, dict):
                        history_dicts.append(msg)
                return format_conversation_history(
                    history_dicts, include_header=False, max_content_length=10000
                ) if history_dicts else "No conversation history"
            return "No conversation history"
        
        try:
            prompts = [self.system_prompt.format(question=item.get('question', ''), planning=item.get('planning', '') or "No planning info",
                    conversation_history=format_history(item.get('conversation_history', ''))) for item in data_list]
            responses = self.client.batch_chat(prompts=prompts, threads=threads, verbose=not is_single, **kwargs)
            # Extract content field
            results = [resp_dict["content"].strip() if isinstance(resp_dict, dict) else str(resp_dict) for resp_dict in responses]
            return results[0] if is_single else results
            
        except Exception as e:
            print(f"Summary error: {e}")
            return "" if is_single else [""] * len(data_list)
    
    def summarize_from_manager(
        self,
        question: str,
        conversation_manager,
        planning: str = "",
        **kwargs
    ) -> str:
        """
        Convenience method to generate summary directly from a ConversationManager object.
        
        Args:
            question: Original question context for summary generation
            conversation_manager: ConversationManager instance
            planning: Planning info (optional)
            
        Returns:
            Compressed summary text
        """
        messages = conversation_manager.get_recent_messages()
        result = self({
            "question": question,
            "conversation_history": messages,
            "planning": planning
        }, **kwargs)
        return result if isinstance(result, str) else result[0]
