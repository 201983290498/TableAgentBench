def parse_assistant_message(messages):
    return {
            "role": "assistant", 
            "content": messages.get("content"), 
            "reasoning_content": messages.get("reasoning_content", None),
            "thought_signature": messages.get("thought_signature", None),
            "tool_calls": messages.get("tool_calls", None)
        }

def filter_reasoning(messages: list, enable_filter: bool = True) -> list:
    """
    Filter reasoning_content from messages if enable_filter is True.
    Handles both dicts and objects.
    """
    if not enable_filter:
        return messages
    # 1. Find the last user conversation message
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg['role'] == 'user' and msg['content'] and not msg['content'].startswith('[Tool Execution Result'):
            last_user_idx = i

    sanitized_messages = []
    for i, msg in enumerate(messages):
        msg_copy = msg.copy()
        if i <= last_user_idx and "reasoning_content" in msg_copy:
            msg_copy['reasoning_content'] = None
            msg_copy['thought_signature'] = None
        sanitized_messages.append(msg_copy) 
    return sanitized_messages