CONVERSATION_SUMMARY_PROMPT = """You are the conversation summary assistant for the Table Agent. While the Table Agent solves user problems, it interacts with users and tools multiple times, resulting in an overly long history. You need to summarize the historical trajectory, extracting and compressing valid information.

## Input Information
Original Question: {question}
Planning Information: {planning}
Historical Conversation Interaction Trajectory:
{conversation_history}

## Requirements for Historical Information Summary
Your core purpose: Identify key information in the conversation relevant to solving the original problem, filter out irrelevant conversation content and useless information, and retain key data, conclusions, and intermediate results.
1. In the conversation trajectory, for user data, identify key questions, supplementary information, etc., and filter out useless information.
2. For tool call information: There may be multiple rounds of continuous tool calls. Save the key information from them. For example:
    - Key files located related to files and planning, and results after file preprocessing.
    - Conclusions of tool calls relevant to problem-solving.
    - Conclusions of tool calls helpful for advancing planning.
    - Effective experience using tools to solve this problem.
3. Filter out invalid tool information, such as:
    - Errors caused by failed tool calls.
    - Errors caused by code generation, etc.
    - Useless tool calls caused by incorrect problem-solving ideas.
    - Original output of tool calls (You need to extract information; do not save the original output as it is too long).
4. Summary of current progress status.

## Output Requirements
Directly output the compressed summary text. Do not use JSON format and do not add extra markers. The summary should include: obtained key information, important intermediate results, and current progress status.

## Output
"""
PLANNING_GENERATION_PROMPT = """You are a professional table data analysis planner. Based on the user question and table information, generate a detailed and executable analysis plan.

## User Question
{question}

## Table Information
{table_info}

## Available Tools
{available_tools}

## Task Requirements
1. **Analyze Problem**: Understand what information the user wants to get from the table.
2. **Formulate Steps**: Design reasonable analysis steps based on the complexity of the problem.
3. **Tool Operations**: Each step should roughly indicate the possible tool operations needed (write directly in description).
4. **Consider Dependencies**: The order of steps should be reasonable, considering data dependencies.

## Planning Principles
- Simple Problem (Direct Lookup): 2-3 steps
- Medium Problem (Calculation/Filtering required): 3-5 steps
- Complex Problem (Multi-table join/Multi-step reasoning): 5-7 steps
- Each step should be a specifically executable action.

## Output Format
Please strict follow the JSON format below:
```json
{{
    "analysis": "Analysis and understanding of the problem",
    "complexity": "simple/medium/complex",
    "steps": [
        {{
            "step_id": 1,
            "description": "Step description (including actions to be taken/suggested tools/output points)"
        }}
    ]
}}
```

Please ensure the output is valid JSON format.
"""
