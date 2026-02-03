import json
import re
import os
import sys


# Add project root to sys.path
# When imported as a package, __file__ may point to site-packages etc., but here we assume it is in the project structure.
# If run directly as a script, we need to add root.
# If imported as src.evaluation, the caller usually has already set up the path.
# For compatibility, retain path check.
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/evaluation/agent -> src/evaluation -> src -> project_root
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.utils.chat_api import ChatClient
from src.utils.common import parse_json_response


class Action:
    def __init__(self, type=None, thought=None, content=None, result=None, turn=None, success=None, data_source=None):
        self.type = type
        self.thought = thought
        self.content = content
        self.result = result
        self.turn = turn
        self.success = success
        self.data_source = data_source


class QaDetail:
    def __init__(self, query=None, answer=None, tool_call_turns=None, is_missing=False, true_answer=None, step_index=None, data_source=None, related_tables=None, score_points=None):
        self.query = query
        self.answer = answer
        self.tool_call_turns = tool_call_turns or []
        self.is_missing = is_missing
        self.true_answer = true_answer
        self.step_index = step_index
        self.data_source = data_source
        self.related_tables = related_tables
        self.score_points = score_points or []


class ToolMetrics:
    def __init__(self, total_tokens=0, tool_success_rate=0.0, tool_parallelism=0.0, tool_calls=None):
        self.total_tokens = total_tokens
        self.tool_success_rate = tool_success_rate
        self.tool_parallelism = tool_parallelism
        self.tool_calls = tool_calls or []


def extract_metadata(data):
    """
    Extract metadata and full trajectory from JSON file.
    """
    metadata = data.get('metadata', {})
    trace_list = data.get('conversation_trace', [])
    last_trace = trace_list[-1] if trace_list else {}
    
    # Construct full message list: messages + response
    if last_trace:
        last_trace = last_trace.copy()
        messages = last_trace.get('messages', [])
        response = last_trace.get('response', {})
        if response:
            # Construct Assistant message, retain tool_calls field to support OpenAI format
            assistant_msg = {'role': 'assistant', 'content': response.get('content', '')}
            if 'tool_calls' in response and response['tool_calls']:
                assistant_msg['tool_calls'] = response['tool_calls']
            messages = messages + [assistant_msg]
            last_trace['messages'] = messages


    return {
        "task": metadata.get('query'),
        "total_turns": metadata.get('total_turns'),
        "success": metadata.get('success'),
        "last_trace": last_trace
    }


def calculate_tool_metrics(last_trace):
    """
    Calculate trajectory metrics: total tokens, tool success rate, tool parallelism.
    """
    if not last_trace:
        return ToolMetrics()
    messages = last_trace.get('messages', [])
    # 1. Calculate total token count
    try:
        total_tokens = ChatClient.count_tokens(messages)
    except Exception as e:
        print(f"Warning: Token count failed: {e}")
        total_tokens = 0
        
    # 2. Calculate tool success rate
    parsed_items = parse_conversation(last_trace)
    tool_actions = [item for item in parsed_items if item.type == 'tool']
    total_tools = len(tool_actions)
    if total_tools > 0:
        success_tools = sum(1 for item in tool_actions if item.success == 'success')
        tool_success_rate = success_tools / total_tools
    else:
        tool_success_rate = 0.0
    
    # Extract tool calls info
    tool_calls_info = []
    for item in tool_actions:
        try:
            content_json = json.loads(item.content)
            tool_name = content_json.get('tool', 'unknown')
            params = content_json.get('params', {})
        except:
            tool_name = item.content
            params = {}
            
        tool_calls_info.append({
            "tool_name": tool_name,
            "params": params,
            "success": item.success,
            "result": item.result,
            "turn": item.turn
        })
        
    # 3. Calculate tool parallelism (number of tools in parsed_items / number of messages containing tool_calls)
    assistant_msgs_with_tools = 0
    for msg in messages:
        # Compatibility check for two formats
        has_openai_tool = msg.get('tool_calls') is not None and len(msg.get('tool_calls')) > 0
        has_legacy_tool = '<tool_call>' in msg.get('content', '')
        if msg.get('role') == 'assistant' and (has_openai_tool or has_legacy_tool):
            assistant_msgs_with_tools += 1
            
    if assistant_msgs_with_tools > 0:
        tool_parallelism = total_tools / assistant_msgs_with_tools
    else:
        tool_parallelism = 0.0
    return ToolMetrics(total_tokens, tool_success_rate, tool_parallelism, tool_calls_info)


def parse_conversation(last_trace):
    """
    Parse conversation trace to extract Query, Tool, and Answer information.
    Automatically identifies and handles OpenAI format vs Legacy format based on message features.
    """
    if not last_trace:
        return []
    
    messages = last_trace.get('messages', [])
    
    # 1. Auto-detect format
    is_openai_format = False
    for msg in messages:
        # If structural tool_calls or role='tool' is found, assume OpenAI format
        if (msg.get('role') == 'assistant' and msg.get('tool_calls')) or msg.get('role') == 'tool':
            is_openai_format = True
            break
            
    if is_openai_format:
        return _parse_openai_format(messages)
    else:
        return _parse_legacy_format(messages)


def _parse_legacy_format(messages):
    """
    Use original code logic to handle Legacy / Tag-based format.
    """
    parsed_items = []
    turn_count = 0
    # Regular expressions
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'system':
            continue
        if role == 'user':
            if '[Tool Execution Result' in content:
                # Split by [Tool Execution Result, each segment is a complete tool result
                parts = content.split('[Tool Execution Result')
                tool_results = []
                for i, part in enumerate(parts):
                    if part.strip():  # Ignore empty strings
                        # The first part is not a tool result (might be empty or other content), subsequent parts are
                        if i > 0:
                            result = '[Tool Execution Result' + part
                            tool_results.append(result.strip())
                
                # Match in order to unfilled tool or unknown items in parsed_items (handles failed tool call parsing)
                for result in tool_results:
                    for item in parsed_items:
                        if (item.type in ['tool', 'unknown']) and item.result is None:
                            item.result = result
                            item.success = 'success' if '[SUCCESS]' in result else 'failed'
                            break
            else:
                turn_count += 1
                parsed_items.append(Action(type="query", turn=turn_count, content=content))
                
        elif role == 'assistant':
            # Detect if specific tags are present (identify type even if parsing fails)
            has_tool_call = '<tool_call>' in content
            has_answer = '<answer>' in content
            
            # Extract tool calls
            tool_calls = tool_call_pattern.findall(content)
            # Extract answers
            answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            answers = answer_pattern.findall(content)
            final_answer = answers[0].strip() if answers else None
            
            # thought keeps raw conversation content without cleaning
            thought = content
            
            # Handle tool calls
            if tool_calls:
                # Successfully parsed tool_call
                for tc in tool_calls:
                    parsed_items.append(Action(type="tool", thought=thought, turn=turn_count, content=tc))
            elif has_tool_call:
                # tool_call tag exists but parsing failed, mark as unknown type
                parsed_items.append(Action(type="unknown", thought=thought, turn=turn_count, content=content))
            
            # Handle answers
            if final_answer:
                # Successfully parsed answer
                parsed_items.append(Action(type="answer", thought=thought, turn=turn_count, content=final_answer))
            elif has_answer:
                # answer tag exists but parsing failed, mark as unknown type
                parsed_items.append(Action(type="unknown", thought=thought, turn=turn_count, content=content))
            elif not has_tool_call and not has_answer:
                # Neither tool_call nor answer tags exist, mark as unknown type
                parsed_items.append(Action(type="unknown", thought=thought, turn=turn_count, content=content))
                
    return parsed_items


def _parse_openai_format(messages):
    """
    Specifically handle OpenAI format.
    - Assistant contains tool_calls field.
    - Tool results are in role='tool' messages.
    - Answer is usually the content, no tags, needs identification via keywords or default logic.
    """
    parsed_items = []
    turn_count = 0
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'system':
            continue
        if role == 'user':
            # In OpenAI format, user message only contains query
            if content and ('[ERROR]' in content or '[Tool Execution Result]' in content):
                continue
            turn_count += 1
            parsed_items.append(Action(type="query", turn=turn_count, content=content))
        elif role == 'tool':
            # Order matching: find the first unfilled Action of type tool
            for item in parsed_items:
                if item.type == 'tool' and item.result is None:
                    # Format result to maintain consistency
                    success_status = '[SUCCESS]' # Default to success
                    if 'Error' in content or 'Exception' in content: 
                         # Simple error detection
                         pass 
                    # Maintain consistency with Legacy format result string for downstream processing
                    item.result = f"[Tool Execution Result]: {content}"
                    item.success = 'success'
                    break
        elif role == 'assistant':
            tool_calls_data = msg.get('tool_calls')
            thought = content
            if tool_calls_data:
                # Handle tool calls
                for tc in tool_calls_data:
                    func = tc.get('function', {})
                    tool_name = func.get('name')
                    tool_args = func.get('arguments')
                    # Construct JSON string compatible with old format
                    try:
                        if isinstance(tool_args, str):
                            params = json.loads(tool_args)
                        else:
                            params = tool_args
                        tool_call_json = json.dumps({"tool": tool_name, "params": params}, ensure_ascii=False)
                    except:
                        tool_call_json = json.dumps({"tool": tool_name, "params": tool_args}, ensure_ascii=False)
                    parsed_items.append(Action(type="tool", thought=thought, turn=turn_count, content=tool_call_json))
            else:
                # 1. Try to find <answer> tag
                answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
                answers = answer_pattern.findall(content)
                final_answer = answers[0].strip() if answers else None
                data_source = []
                if not final_answer:
                    json_data = parse_json_response(content)
                    if isinstance(json_data, dict) and "error" not in json_data and 'answer' in json_data:
                        ans = json_data['answer']
                        if isinstance(ans, str):
                            final_answer = ans.strip()
                        else:
                            final_answer = json.dumps(ans, ensure_ascii=False)
                        data_source = json_data.get('data_source', [])
                
                if not final_answer:
                    # 3. Try to find keywords
                    keywords = ["**Answer:**", "**Answer:**", "## Answer", "### Answer", "## Response", "### Response", "Answer:"]
                    for kw in keywords:
                        if kw in content:
                            parts = content.split(kw, 1)
                            if len(parts) > 1:
                                final_answer = parts[1].strip()
                                break
                
                # 4. Handle final result
                if final_answer:
                    parsed_items.append(Action(type="answer", thought=thought, turn=turn_count, content=final_answer, data_source=data_source))
                else:
                    if content and content.strip():
                        parsed_items.append(Action(type="answer", thought=thought, turn=turn_count, content=content, data_source=data_source))
                    else:
                        parsed_items.append(Action(type="unknown", thought=thought, turn=turn_count, content="Empty Content", data_source=data_source))
    return parsed_items

def analyze_query_answer_pairs(parsed_items, eval_info_list=None):
    """
    Match Query and Answer, including intermediate tool calls.
    If eval_info_list is provided, performs forced alignment and marks is_missing based on eval_info_list.
    """
    # 1. First pass: extract raw pairs from trace
    raw_pairs = []
    current_query = None
    current_tools = []
    
    for item in parsed_items:
        if item.type == 'query':
            if current_query:
                raw_pairs.append(QaDetail(
                    query=current_query,
                    answer=None,
                    tool_call_turns=current_tools,
                    step_index=len(raw_pairs) + 1
                ))
            current_query = item.content
            current_tools = []
        elif item.type in ['tool', 'unknown']:
            if current_query:
                current_tools.append(item)
        elif item.type == 'answer':
            if current_query:
                raw_pairs.append(QaDetail(current_query, item.content, current_tools,step_index=len(raw_pairs)+1, data_source=item.data_source))
                current_query = None
                current_tools = []
                
    if current_query:
        raw_pairs.append(QaDetail(current_query, None, current_tools, step_index=len(raw_pairs)+1, data_source=[]))
    if eval_info_list is None:
        return raw_pairs
    aligned_pairs = []
    for i, info in enumerate(eval_info_list):
        qa_detail = QaDetail(info.get('info_item', ''), "", [], True, info.get('answer', ''), i + 1, [], info.get('related_tables', []), info.get('score_points', []))
        if i < len(raw_pairs):
            qa_detail.tool_call_turns = raw_pairs[i].tool_call_turns
            if raw_pairs[i].answer:
                qa_detail.answer = raw_pairs[i].answer
                qa_detail.is_missing = False
                qa_detail.data_source = [item.split("/")[-1] for item in raw_pairs[i].data_source]
        aligned_pairs.append(qa_detail)
    return aligned_pairs


def get_eval_info(query, eval_file_path):
    """
    Retrieve evaluation samples from the evaluation file based on the query, link and extract fields like info_item, related_tables, answer, etc.
    """
    if not os.path.exists(eval_file_path):
        print(f"Warning: Evaluation file does not exist: {eval_file_path}")
        return None
    try:
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            eval_cases = json.load(f)
        # Normalize query string (strip whitespace)
        normalized_query = query.strip() if query else ""
        for case in eval_cases:
            # Compatibility for task or question fields
            case_task = case.get('task', '') or case.get('question', '')
            case_task = case_task.strip()
            # Try exact match or inclusion match
            if case_task == normalized_query or (normalized_query and normalized_query in case_task) or (case_task and case_task in normalized_query):
                # Mode 1: Complex evaluation (contains design.checkout_list)
                if 'design' in case and 'checkout_list' in case['design']:
                    checkout_list = case['design']['checkout_list']
                    extracted_info = []
                    for item in checkout_list:
                        extracted_info.append({
                            "idx": item.get("idx"),
                            "info_item": item.get("info_item"),
                            "related_tables": item.get("related_tables", []),
                            "answer": item.get("answer"),
                            "score_points": item.get("score_points",[])
                        })
                    return extracted_info
                # Mode 2: Simple evaluation (directly contains answer)
                elif 'answer' in case:
                    return [{
                        "idx": 1,
                        "info_item": case_task,
                        "related_tables": case.get('table_path', []) if isinstance(case.get('table_path'), list) else [case.get('table_path', '')],
                        "answer": str(case['answer']),
                        "score_points": case.get("score_points", [])
                    }]
        return None
        
    except Exception as e:
        print(f"Error reading eval file: {e}")
        return None

