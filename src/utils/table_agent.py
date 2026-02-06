from typing import Optional

import re
from typing import Optional

def validate_response_format(response: str) -> tuple[bool, Optional[str]]:
    """Validate if the LLM response format is correct
    
    Returns:
        (is_valid, error_message): Whether the format is valid, and the error message
    """
    errors, warnings = [], []
    
    # 0. Pre-processing: Unified tags
    response_norm = response.replace('<thinking>', '<think>').replace('</thinking>', '</think>')
    
    # 1. Check think tags (warning only)
    think_open = response_norm.count('<think>')
    think_close = response_norm.count('</think>')
    
    if think_open == 0:
        warnings.append("Missing thinking tags <think>...</think>")
    elif think_open != think_close:
        warnings.append(f"<think> tags mismatch: {think_open} open, {think_close} closed")
        
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    content_without_think = think_pattern.sub('', response_norm)
    
    if think_open > think_close:
        pass 
    
    # 3. Check action tags in the content after removing think tags
    
    has_tool_call = '<tool_call>' in content_without_think
    has_answer = '<answer>' in content_without_think
    
    if not has_tool_call and not has_answer:
        errors.append("Missing action tags: requires <tool_call>...</tool_call> or <answer>...</answer>")
    
    # Check tool_call tag pairing (required)
    if has_tool_call:
        tool_call_open = content_without_think.count('<tool_call>')
        tool_call_close = content_without_think.count('</tool_call>')
        if tool_call_open != tool_call_close:
            errors.append(f"<tool_call> tags mismatch: {tool_call_open} open, {tool_call_close} closed")
    
    # Check answer tag pairing (required)
    if has_answer:
        answer_open = content_without_think.count('<answer>')
        answer_close = content_without_think.count('</answer>')
        if answer_open != answer_close:
            errors.append(f"<answer> tags mismatch: {answer_open} open, {answer_close} closed")
    
    # Check current_step tag pairing (required if it exists)
    if '<current_step>' in content_without_think:
        current_step_open = content_without_think.count('<current_step>')
        current_step_close = content_without_think.count('</current_step>')
        if current_step_open != current_step_close:
            errors.append(f"<current_step> tags mismatch: {current_step_open} open, {current_step_close} closed")
    
    if errors:
        return False, "[ERROR] Output format error:\n" + "\n".join(f"- {e}" for e in errors)
    
    return True, None
