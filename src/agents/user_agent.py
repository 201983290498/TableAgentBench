from typing import List, Dict, Optional, Any
import json
from src.utils.chat_api import ChatClient
from dataclasses import dataclass
from src.prompts.UserAgentPrompt import USER_Agent_PROMPT

@dataclass
class UserAgentConfig:
    model_provider: str = "azure"  # Model provider, default Azure
    model_name: str = "gpt-4o"      # Model name, default gpt-4o
    config_key: str = "gpt-4o"      # Config key, default same as model name

class UserAgent:
    """
    Simulate user in multi-turn conversation for table analysis tasks.
    """
    def __init__(self, config: UserAgentConfig, enable_thinking=False):
        # Initialize LLM client
        self.llm_client = ChatClient(config_key=config.config_key)
        self.task_data = None          # Current task data
        self.current_step_idx = 0      # Current checklist item index
        self.history = []              # Conversation history
        self.enable_thinking = enable_thinking
        
    def reset(self, task_data: Dict[str, Any]):
        """
        Reset agent with new task.
        
        Args:
            task_data: Task data item from all_correct_items.json
        """
        self.task_data = task_data
        self.current_step_idx = 0
        self.history = []
        
    def get_system_prompt(self) -> str:
        """Generate system prompt based on current task."""
        task_desc = self.task_data.get("task", "")
        checkout_list = self.task_data.get("design", {}).get("checkout_list", [])
        checklist_str = "\n".join([
            f"{item['idx']}. {item['info_item']}" 
            for item in checkout_list
        ])
        prompt = USER_Agent_PROMPT.format(task_desc=task_desc, checklist_str=checklist_str, current_step_idx=self.current_step_idx)
        return prompt

    def generate_question(self, last_answer: Optional[str] = None) -> Optional[str]:
        """
        Generate next question based on history and current goal.
        
        Args:
            last_answer: The last answer from the assistant, optional.
        
        Returns:
            The generated question string; returns None if the task is finished.
        """
        checkout_list = self.task_data.get("design", {}).get("checkout_list", [])
        if self.current_step_idx >= len(checkout_list):
            return None

        # If the last answer is provided, update history
        if last_answer:
            self.history.append({"role": "assistant", "content": last_answer})
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        messages.extend(self.history)
        current_item = checkout_list[self.current_step_idx]
        user_instruction = f"Please generate the next question for item {self.current_step_idx} in the Checklist: '{current_item['info_item']}'."
        messages.append({"role": "user", "content": user_instruction})
        
        # Call LLM
        # response = self.llm_client.batch_chat(
        #     messages=[messages],
        #     enable_thinking=self.enable_thinking,
        # )[0]
        # self.history.append({"role": "user", "content": response["content"].strip()})
        # self.current_step_idx += 1
        # return response["content"].strip()

        # Directly output the checkpoint
        question = current_item['info_item']
        self.history.append({"role": "user", "content": question})
        self.current_step_idx += 1
        return question

    def is_finished(self) -> bool:
        """Check if the task is finished."""
        checkout_list = self.task_data.get("design", {}).get("checkout_list", [])
        return self.current_step_idx >= len(checkout_list)
