"""
Conversation History Management - Simplified version
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.function_llm import ConversationSummaryLLM
from src.utils.chat_api import ChatClient

class ConversationManager:
    """
    Conversation History Manager - Simplified version
    Responsibility: Store conversations, manage context length, generate summaries.
    """
    
    def __init__(
        self, 
        max_messages: int = 40,
        min_messages: int = 10,
        max_tokens: int = 4000,
        enable_summary: bool = False,
        summarizer: "ConversationSummaryLLM" = None,
    ):
        """
        Args:
            max_messages: Maximum number of messages to trigger trimming.
            min_messages: Number of messages to retain after trimming.
            max_tokens: Maximum number of tokens to trigger trimming.
            enable_summary: Whether to enable summary.
            summarizer: Summary generator.
        """
        self.max_messages = max_messages
        self.min_messages = min_messages
        self.max_tokens = max_tokens
        self.enable_summary = enable_summary
        self.summarizer = summarizer
        
        # State
        self.messages: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.question: str = ""
        self.planning: str = ""
    
    def set_question(self, question: str):
        """Set current question."""
        self.question = question
    
    def set_planning(self, planning: str):
        """Set planning information."""
        self.planning = planning
    
    def add_message(self, role: str, content, is_tool_result: bool = False):
        """Add message."""
        if type(content) == str:
            content = {"content": content}
        self.messages.append({
            "role": role, 
            **content,
            "is_tool_result": is_tool_result
        })
        self._trim_if_needed()
    
    def add_tool_result(self, tool_name: str, result: str, tool_id=None):
        # Check tool result length (estimate tokens)
        result_tokens = ChatClient.count_tokens(result)
        if result_tokens > 2000:
            import warnings
            warnings.warn(f"[ConversationManager] Tool '{tool_name}' result too long: {result_tokens} tokens "
                f"(suggest <2000), which may cause context overflow. Suggest compressing the result inside the tool.", UserWarning,stacklevel=2)

        if tool_id is None:
            """Add tool execution result (consecutive tool results will be merged into one message)."""
            
            new_content = f"[Tool Execution Result: {tool_name}]\n{result}"
            # If the previous message is also a tool result, merge into the same message
            if self.messages and self.messages[-1].get('is_tool_result', False):
                self.messages[-1]['content'] += f"\n\n{new_content}"
                self._trim_if_needed()  # Check for trimming after merging
            else:
                self.add_message(
                    role="user",
                    content=new_content,
                    is_tool_result=True
                )
        else:
            self.add_message(
                role="tool",
                content={"content": result, "tool_call_id": tool_id},
                is_tool_result=True
            )
            
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get message list (for LLM)."""
        messages = []
        # 1. Filter out leading tool messages (defensive programming, prevent orphan tool messages from causing API errors)
        start_idx = 0
        while start_idx < len(self.messages) and self.messages[start_idx]['role'] == 'tool':
            start_idx += 1
        for i in range(start_idx, len(self.messages)):
            m = self.messages[i]
            if m['role'] == 'assistant':
                msg = {
                    "role": m["role"], 
                    "content": m["content"], 
                    "reasoning_content": m.get("reasoning_content", None),
                    "thought_signature": m.get("thought_signature", None)
                }
                if m.get("tool_calls"):
                    msg["tool_calls"] = m.get("tool_calls")
                messages.append(msg)
            elif m['role'] == 'tool':
                if m.get('tool_call_id'):
                    messages.append({"role": m["role"], "content": m["content"], 'tool_call_id': m['tool_call_id']})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        return messages
    
    def get_summary(self) -> str:
        """Get summary."""
        return self.summary
    
    def _trim_if_needed(self):
        """Check and trim conversation history if needed."""
        tokens = self._estimate_tokens()
        msg_count = len(self.messages)
        
        if msg_count >= self.max_messages or tokens >= self.max_tokens:
            if self.enable_summary and self.summarizer:
                self._summarize_and_trim()
            else:
                self._simple_trim()
    
    def _estimate_tokens(self) -> int:
        """Estimate token count."""
        return ChatClient.count_tokens(self.messages)
    
    def _simple_trim(self):
        """Simple trimming (keep most recent messages)."""
        removed_messages = []
        
        if len(self.messages) > self.max_messages:
            removed_count = len(self.messages) - self.max_messages
            removed_messages.extend(self.messages[:removed_count])
            self.messages = self.messages[removed_count:]
            
            # Ensure the list doesn't start with a tool message (tool messages must follow assistant).
            while self.messages and self.messages[0].get('role') == 'tool':
                removed_messages.append(self.messages.pop(0))
        
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 1:
            removed_messages.append(self.messages.pop(0))
            # Ensure the remaining list doesn't start with a tool message.
            while self.messages and self.messages[0].get('role') == 'tool':
                removed_messages.append(self.messages.pop(0))
        
        # Handle signature preservation.
        if removed_messages:
            self._preserve_signature(removed_messages)
    
    def _summarize_and_trim(self):
        """Summarize and trim."""
        if len(self.messages) <= self.min_messages:
            return
        
        old_messages = self.messages[:-self.min_messages]
        recent_messages = self.messages[-self.min_messages:]
        
        # If recent_messages starts with a tool, it means it's truncated, move back to old_messages for summary.
        while recent_messages and recent_messages[0].get('role') == 'tool':
            old_messages.append(recent_messages.pop(0))
        
        if old_messages and self.summarizer:
            new_summary = self.summarizer({
                "question": self.question,
                "conversation_history": old_messages,
                "planning": self.planning
            })
            if new_summary:
                self.summary = f"{self.summary}\n---\n{new_summary}" if self.summary else new_summary
        
        self.messages = recent_messages
        
        # Handle signature preservation.
        self._preserve_signature(old_messages)
    
    def _preserve_signature(self, removed_messages: List[Dict[str, Any]]):
        """Preserve thought_signature from removed messages."""
        if not removed_messages:
            return

        # 1. Find the latest thought_signature in removed messages.
        target_signature = None
        for msg in reversed(removed_messages):
            if msg.get("thought_signature"):
                target_signature = msg.get("thought_signature")
                break
        if not target_signature:
            return

        # 2. Check if remaining messages already contain a thought_signature.
        has_signature = any(msg.get("thought_signature") for msg in self.messages)
        
        if not has_signature:
            # 3. Attach the signature to the earliest assistant message in remaining messages.
            for msg in self.messages:
                if msg.get("role") == "assistant":
                    msg["thought_signature"] = target_signature
                    break

    
    def clear(self, keep_summary: bool = False):
        """Clear messages."""
        self.messages = []
        if not keep_summary:
            self.summary = ""
    
    def reset_for_new_question(self, question: str, keep_summary: bool = True):
        """Reset for a new question."""
        self.clear(keep_summary=keep_summary)
        self.question = question
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "message_count": len(self.messages),
            "has_summary": bool(self.summary),
            "estimated_tokens": self._estimate_tokens()
        }
