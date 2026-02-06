"""
TableAgent Core Class
Planning-driven Table QA Agent
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json
import re
from enum import Enum
import os
from datetime import datetime
from src.agents.context.context_manager import ContextManager
from src.agents.env_manager import EnvManager
from src.tools import get_tool, get_tools_schema
from src.tools.base import ToolResult
from src.prompts.agent_prompts import AGENT_SYSTEM_PROMPT_SIMPLE_FINAL
from src.utils.chat_api import ChatClient
from src.function_llm import ConversationSummaryLLM
from src.utils.global_config import GLOBAL_CONFIG

class AgentState(Enum):
    """Agent State"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """Agent Action"""
    action_type: str  # "tool", "answer", "error"
    tool_name: Optional[str] = None
    tool_params: Optional[Dict] = None
    answer: Optional[str] = None
    thinking: Optional[str] = None
    tool_id: Optional[str] = None  # Tool call ID, for auto parsing


@dataclass
class AgentOutput:
    """Agent Output"""
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    conversation_trace: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0


class TableAgent:
    """
    TableAgent
    Planning-driven Table QA Agent, flexible tool usage by type
    """
    
    # Parse Regex
    THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    def __init__(
        self,
        llm_client: Optional[ChatClient] = None,
        planning_llm: Optional[Any] = None,
        verbose: bool = True,
        max_steps: int = 70,
        max_context_messages: int = 800,
        min_context_messages: int = 10,
        max_history_tokens: int = 16384,
        summarizer: Optional[ConversationSummaryLLM] = None,
        enable_thinking: bool = True,
        trace_save_dir: Optional[str] = None,
        multi_turn_mode: bool = False,  # Multi-turn mode: accumulate trace, delay save
        reset_env: bool = False,  # Whether to reset environment after each session
        include_tools: Optional[List[str]] = None,
        save_sft: bool = False, # 是否保存用于sft的数据
    ):
        self.llm_client = llm_client
        self.planning_llm = planning_llm
        self.verbose = verbose
        self.max_steps = max_steps
        self.enable_thinking = enable_thinking
        self.include_tools = include_tools
        self.save_sft = save_sft
        self.sft_data = []  # 用于保存SFT数据的列表
        self.multi_turn_mode = multi_turn_mode
        self._is_first_turn = True  # Flag for first turn of session
        self.step_count = 0
        self.env_manager = EnvManager() if reset_env else None
        self.trace_save_dir = trace_save_dir
        self.conversation_trace: List[Dict[str, Any]] = []  # Multi-turn conversation trace for SFT
        self.context = ContextManager(
            max_messages=max_context_messages,
            min_messages=min_context_messages,
            max_tokens=max_history_tokens,
            summarizer=summarizer,
            system_prompt_template=AGENT_SYSTEM_PROMPT_SIMPLE_FINAL,
        )
        self._reset_execution_state()
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _restore_env(self) -> int:
        """Restore file environment (internal method)"""
        if not self.env_manager or not self.env_manager._snapshot_taken:
            return 0
        deleted = self.env_manager.restore(verbose=self.verbose)
        if self.verbose and deleted > 0:
            self._log(f"[EnvManager] Environment restored: deleted {deleted} files")
        self.env_manager.reset()
        return deleted

    def run(
        self, query: str, table_path: str = "",
        continue_mode: bool = False,
        keep_summary: bool = False
    ) -> AgentOutput:
        """
        Unified run entrance
        
        Args:
            query: User question
            table_path: Table path
            continue_mode: Follow-up mode (keep full conversation history, reset Planning)
            keep_summary: Keep history summary mode (only keep summary, reset Planning and conversation)
        """
        self._log(f"[TableAgent] Processing: {query}")
        # Environment management (only for the first turn and when environment management is enabled)
        if self.env_manager and self._is_first_turn and table_path:
            # Multi-turn mode: restore old environment before new session (if any)
            if self.multi_turn_mode:
                self._restore_env()
            # Create new snapshot
            file_count = self.env_manager.snapshot(table_path)
            self._log(f"[EnvManager] Environment snapshot: {file_count} files")
        # Multi-turn mode: automatically determine continue_mode
        if self.multi_turn_mode:
            continue_mode = not self._is_first_turn
            self._is_first_turn = False
        # Unified initialization: Planning generation and execution state reset shared by three modes
        self._init_session(
            query=query, 
            table_path=table_path, 
            continue_mode=continue_mode,
            keep_summary=keep_summary
        )
        self.state = AgentState.RUNNING
        return self._main_loop()
    
    def _init_session(
        self, 
        query: str, 
        table_path: str,
        continue_mode: bool = False,
        keep_summary: bool = False
    ):
        """
        Initialize session
        
        Args:
            query: User question
            table_path: Table path
            continue_mode: Follow-up mode - keep full conversation history
            keep_summary: Summary mode - only keep summary
        """
        # 2. Initialize context based on mode
        if continue_mode:
            # Follow-up mode: keep full conversation history, only reset Planning
            self.context.continue_with_new_question(
                new_query=query,
            )
        elif keep_summary:
            # Summary mode: keep summary, reset conversation and Planning
            self.context.update_for_new_question(
                new_query=query,
                keep_summary=True
            )
        else:
            # Complete reset mode
            self.context.init_session(
                query=query, 
                table_path=table_path, 
            )
        
        if table_path:
            self.context.current_table_path = table_path
        # 3. Reset execution state (required for all modes)
        self._reset_execution_state()
    
    def _main_loop(self) -> AgentOutput:
        """Main execution loop"""
        final_answer = None
        
        while self.step_count < self.max_steps:
            self.step_count += 1
            self._log(f"******\n[Step {self.step_count}]")
            
            # Get and parse response
            response = self._get_llm_response()
            self._log(f"[LLM] Response: {response}")
            actions = self._parse_response(response)
            # Execute actions
            for action in actions:
                result = self._execute_action(action)
                if result:
                    self._log(f"[Result] {result}")
                if action.action_type == "answer":
                    final_answer = action.answer
                if self.state in [AgentState.COMPLETED, AgentState.FAILED]:
                    break
            if self.state == AgentState.COMPLETED:
                return self._make_output(True, answer=final_answer)
            if self.state == AgentState.FAILED:
                return self._make_output(False, error="Execution failed")
        return self._make_output(False, error=f"Exceeded max steps({self.max_steps})")
    
    def _make_output(self, success: bool, answer: str = None, error: str = None) -> AgentOutput:
        """Construct output"""
        # Single-turn mode: automatically restore environment
        if self.env_manager and not self.multi_turn_mode:
            self._restore_env()
        # Save conversation trace (not automatically saved in multi-turn mode, wait for external save_trace call)
        if self.trace_save_dir and not self.multi_turn_mode:
            self._save_trace_to_file()
        return AgentOutput(
            success=success,
            answer=answer,
            error=error,
            conversation_trace=self.conversation_trace,
            total_steps=self.step_count
        )

    # ============================================================
    # LLM Interaction
    # ============================================================
    
    def _get_llm_response(self,) -> str:
        """Get LLM response"""
        messages = self.context.build_messages()
        current_tools_schema = get_tools_schema(include_tools=self.include_tools)
        response_dict = self.llm_client.chat(message=messages, enable_thinking=self.enable_thinking, tools=current_tools_schema, temperature=0.0, seed=42)  
        
        content = response_dict['content']
        # 1. Unify tags: <thinking> -> <think>; remove isolated closing tags (when reasoning_content exists, content shouldn't have </think>); remove isolated </think> (without corresponding <think>)
        if '<thinking>' in content or '</thinking>' in content:
            content = content.replace('<thinking>', '<think>').replace('</thinking>', '</think>')
        if response_dict['reasoning_content'] and '</think>' in content:
            content = content.replace('</think>', '')
        if '<think>' not in content and '</think>' in content:
            content = content.replace('</think>', '')
        raw = ""
        if response_dict['reasoning_content']:
            raw += f"<think>\n{response_dict['reasoning_content'] or ''}\n</think>\n\n"
        if response_dict['tool_calls'] in [[], None]:
            raw += f"<answer> {content or ''}</answer>"
        else:
            raw += content or ""
        if "<tool_call>" not in response_dict['content'] and response_dict['tool_calls']:
            raw += "\n\n"
            for tool_call in response_dict['tool_calls']:
                tool_info = json.dumps({"tool": tool_call['function']['name'], 
                                        "params": tool_call['function']['arguments'],
                                        "call_id": tool_call['id']}, ensure_ascii=False) 
                raw += "<tool_call>" + tool_info + "</tool_call>\n"    
        self.context.add_message("assistant", response_dict)
        self._record_turn(messages, response_dict)
        self._log(f"total: {response_dict['usage']['total_tokens']}; input: {response_dict['usage']['prompt_tokens']}; output: {response_dict['usage']['completion_tokens']}")
        return raw
    # ============================================================
    # Response Parsing
    # ============================================================
    
    def _parse_response(self, response: str) -> List[AgentAction]:
        """Parse LLM response"""
        actions = []
        
        # 1. Normalize tags
        response_norm = response.replace('<thinking>', '<think>').replace('</thinking>', '</think>')
        
        # 2. Extract thinking (take the first match think block as thinking)
        think_match = self.THINK_PATTERN.search(response_norm)
        thinking = think_match.group(1).strip() if think_match else None
        
        # 3. Remove all thinking content, only parse actions in remaining content
        content_without_think = self.THINK_PATTERN.sub('', response_norm)
        
        # 5. Prioritize checking answer
        answer_match = self.ANSWER_PATTERN.search(content_without_think)
        if answer_match:
            return [AgentAction(action_type="answer", answer=answer_match.group(1).strip(), thinking=thinking)]
        
        # 6. Parse tool calls
        for content in self.TOOL_CALL_PATTERN.findall(content_without_think):
            if action := self._parse_tool_action(content, thinking if not actions else None):
                actions.append(action)
        
        return actions or [AgentAction(action_type="error", thinking=thinking)]
    
    def _parse_tool_action(self, content: str, thinking: Optional[str]) -> Optional[AgentAction]:
        """Parse tool call"""
        try:
            content = content.strip()
            try:
                action_json = json.loads(content)
            except json.JSONDecodeError:
                action_json, _ = json.JSONDecoder().raw_decode(content)
            
            return AgentAction(
                action_type="tool",
                tool_name=action_json.get("tool"),
                tool_params=action_json.get("params", {}),
                tool_id = action_json.get("call_id", None),
                thinking=thinking
            )
        except json.JSONDecodeError:
            self._log(f"[Parse] JSON parse failed: {content[:50]}...")
            return AgentAction(action_type="error", thinking=thinking)

    # ============================================================
    # Action Execution
    # ============================================================
    
    def _execute_action(self, action: AgentAction) -> Optional[str]:
        """Execute action"""
        if action.action_type == "answer":
            self.state = AgentState.COMPLETED
            return action.answer
        
        if action.action_type == "tool":
            if type(action.tool_params) == str:
                action.tool_params = json.loads(action.tool_params)
            return self._execute_tool(action.tool_name, action.tool_params or {}, action.tool_id)
        
        if action.action_type == "error":
            self.context.conversation.add_tool_result("system", "[ERROR] Failed to parse response, please use the correct format", action.tool_id)
            return None
        
        return None
    
    def _execute_tool(self, tool_name: str, params: Dict, tool_id: Optional[str] = None) -> str:
        """Execute tool"""
        self._log(f"[Tool] {tool_name}, params: {params}")
        
        tool = get_tool(tool_name)
        if tool is None:
            error_msg = f"[ERROR] Tool '{tool_name}' does not exist"
            self.context.conversation.add_tool_result(tool_name, error_msg, tool_id)
            return error_msg
        
        try:
            result: ToolResult = tool.execute(**params)
            result_str = "[SUCCESS] " + str(result.data) if result.success else "[ERROR] " + result.message
            # Limit tool result length
            max_tool_response = int(GLOBAL_CONFIG.get("table_agent", {}).get("max_tool_response", 3000))
            if len(result_str) > max_tool_response:
                result_str = f"[Truncated leading part, original length: {len(result_str)}] ...\n" + result_str[-max_tool_response:]
            self.context.conversation.add_tool_result(tool_name, result_str, tool_id)
            return result_str
        except Exception as e:
            error_msg = f"[ERROR] Tool execution error: {str(e)}"
            self.context.conversation.add_tool_result(tool_name, error_msg, tool_id)
            return error_msg

    # ============================================================
    # Recording and Utils
    # ============================================================
    
    def _record_turn(self, messages: List[Dict[str, str]], response_dict: Dict[str, Any]):
        last_trace = self.conversation_trace[-1] if self.conversation_trace else None
        response = {
            "content": response_dict.get("content", ""),
            "reasoning_content": response_dict.get("reasoning_content", ""),
            "model": response_dict.get("model", ""),
            "usage": response_dict.get("usage", {}),
            "tool_calls": response_dict.get("tool_calls", []),
            "finish_reason": response_dict.get("finish_reason", None)
        }
        if self.save_sft:
            self.generate_sft_data(messages, response_dict)
        if last_trace is None:
            self.conversation_trace.append({
                "turn": self.step_count,
                "timestamp": datetime.now().isoformat(),
                "messages": messages,
                "response": response
            })
        else:
            idx = -1
            for i, msg in enumerate(messages):
                if json.dumps({"content": msg['content'], "tool_calls": msg.get("tool_calls", [])}, ensure_ascii=False) == \
                    json.dumps({"content": last_trace['messages'][-1]['content'], "tool_calls": last_trace['messages'][-1].get("tool_calls", [])}, ensure_ascii=False):
                    idx = i
                    break
            if idx == -1:
                print('Error message adding.')
            if last_trace['messages'][0]['content'] == messages[0]['content']: # System prompts are the same
                last_trace['messages'].extend(messages[idx+1:])
                self.conversation_trace[-1]['response'] = response
            else:
                old_messages = last_trace['messages'][:]
                old_messages[0]['content'] = messages[0]['content']
                self.conversation_trace.append({
                    "turn": self.step_count,
                    "timestamp": datetime.now().isoformat(),
                    "messages": old_messages + messages[idx+1:],
                    "response": response
                })
        self.conversation_trace[-1]['messages'][-1]['tag'] = {
            "window": len(messages)-1, 
            "start":messages[1]['content'] or messages[1]['reasoning_content'], 
            "completion_tokens": response_dict['usage']['completion_tokens'], 
            "prompt_tokens": response_dict['usage']['prompt_tokens'],
            "total_tokens": response_dict['usage']['total_tokens'],
        }
                   
    def _save_trace_to_file(self, custom_path: Optional[str] = None, task: Optional[str] = None):
        """Save conversation trace to file (for SFT training)"""
        if not self.conversation_trace:
            return
        # Calculate and print token usage
        total_tokens, total_input_tokens, total_output_tokens = 0,0,0
        for msg in self.conversation_trace[-1].get('messages', []):
            if 'tag' in msg:
                total_tokens += msg['tag'].get('total_tokens', 0)
                total_input_tokens += msg['tag'].get('prompt_tokens', 0)
                total_output_tokens += msg['tag'].get('completion_tokens', 0)
        save_dir = custom_path or self.trace_save_dir
        os.makedirs(save_dir, exist_ok=True)
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_short = self.context.current_query[:30].replace(" ", "_").replace("/", "_") if self.context.current_query else "unknown"
        filepath = os.path.join(save_dir, f"trace_{timestamp}_{query_short}.json")
        # Construct save data
        save_data = {
            "metadata": {
                "query": task if task else self.context.current_query,
                "table_path": self.context.current_table_path,
                "total_turns": self.step_count,
                "timestamp": datetime.now().isoformat(),
                "success": self.state == AgentState.COMPLETED
            },
            "conversation_trace": self.conversation_trace,
            "sft_data": self.sft_data if self.save_sft else None
        }
        def default_serializer(obj):
            if isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=default_serializer)
            self._log(f"[Save] Conversation trace saved to: {filepath}")
        except Exception as e:
            self._log(f"[Save] Save failed: {e}")

    def _reset_execution_state(self):
        """
        Reset execution state (internal method, called at the start of each run())
        
        Description:
        - Single-turn mode: reset step_count and trace at each run()
        - Multi-turn mode: keep accumulation in multi-turn conversations of the same sample (for overall sample statistics)
        """
        self.state = AgentState.IDLE
        # Single-turn mode: reset at each run()
        # Multi-turn mode: keep accumulation within the same sample
        if not self.multi_turn_mode:
            self.step_count = 0
            self.conversation_trace = []
    
    def reset(self, keep_summary: bool = False):
        """Reset Agent"""
        self._reset_execution_state()
        self.context.clear(keep_summary=keep_summary)
    
    def save_session_trace(self, custom_path: str = None, task: Optional[str] = None):
        """Manual trace save (used in multi-turn mode)"""
        self._save_trace_to_file(custom_path, task)
    
    def reset_session(self):
        """
        Reset session state (used for sample switching, completely isolating different samples)
        Call this method before each sample starts. Whether the sample is single-turn or multi-turn, it will be completely reset. 
        Multi-turn conversations within a sample are automatically managed via multi_turn_mode.
        """
        # Restore environment (delete newly generated files)
        self._restore_env()
        
        # Completely reset all states (isolation between samples)
        self.step_count = 0
        self.state = AgentState.IDLE
        self._is_first_turn = True
        self.conversation_trace = []
        self.context.clear()

    def generate_sft_data(self, messages: List[Dict[str, str]], response_dict: Dict[str, Any]):
        """Generate SFT data from the current turn and append to sft_data list"""
        new_messages = messages.copy() + [{"role": "assistant", **response_dict}]
        
        # When existing messages are a prefix of new_essages, remove existing messages
        self.sft_data = [
            exist_messages for exist_messages in self.sft_data
            if not (len(exist_messages) <= len(new_messages) and 
                    new_messages[:len(exist_messages)] == exist_messages)
        ]
        self.sft_data.append(new_messages)

def create_table_agent(**kwargs) -> TableAgent:
    """Create TableAgent"""
    return TableAgent(**kwargs)
