from langchain_classic.prompts import PromptTemplate

message_analyzer_p = PromptTemplate(
    input_variables=["message"],
    template="""
    You are a message analyzer and editor.
    Edit message only if:
    -need fixing of grammar and spelling
    -if message is incomplete try to deduce what it means then add to it or rewrite if necessary

    Analyze message and determine:
    -actual message only if it needed editing     
    -message type (QUESTION, COMMAND, INSTRUCTION, STATEMENT)                                                                                               
    -what is being asked or meaning of the message? what does the user want?
    -message complexity (LOW, MEDIUM or HIGH)
    -external data dependency - is it explicitly/implicitly dependent on a document.
    -tool dependency (NONE, LOW, MEDIUM, HIGH) and (tools needed)
    -cognitive operations required, atmost 5 (e.g., comparison, synthesis, planning)
                                    
    message: {message}
                                                    
    Analysis should be your reasoning about the message
    should be short (one to two paragraph long) compiling  all relevant information.
    Output strictly in JSON format:
    {{
        "analysis": "..."
    }}
    """
)

router_p = PromptTemplate(
    input_variables=["messages"],
    template="""
    You are an intelligent task router for an AI agent.
    Given the conversation so far:
    {messages}

    Decide how to handle the latest user request.
    Available routes:
    1) DIRECT
    - Answer immediately using reasoning
    - No tools or external data needed
    2) REACT
    - Solve step-by-step using tools or retrieval
    - Next action is clear at each step
    3) PLAN
    - Break problem into multiple steps
    - Requires decomposition before execution

    Rules:
    - Prefer DIRECT if the question is simple and fully answerable
    - Use REACT if tools or retrieval are needed in a step-by-step way
    - Use PLAN if the task requires multiple steps or coordination
    - Default to PLAN only if REACT is insufficient

    Output STRICT JSON only:
    {{
        "route": "DIRECT | REACT | PLAN",
        "reason": "short explanation",
        "confidence": 0.0-1.0
    }}
    """
)

agent_template_p = PromptTemplate(
    input_variables=["tools", "messages"],
    template="""
    You are an AI agent that follows (Reason → Act → Observe).

    Messages:
    {messages}

    You have access to the following tools:
    {tools}

    When performing task
    1. Think step-by-step about what to do.
    2. Decide if a tool is needed.
    3. If tool needed, fill up action and args in json output.
    4. After each action, you will receive an observation.
    5. Continue reasoning and acting until you can provide a final answer.

    Rules:
    - Do not skip reasoning.
    - Always use tools when external information is required.
    - Do not hallucinate tool outputs.
    - Stop when sufficient information is gathered.
    - Output the final answer in a clear, concise format.

    Output in JSON Format:
    {{
        "thought": "Your reasoning for choosing action",
        "action": "tool_name_here",
        "args": {{
            "arg1": value,
            "arg2": value
        }},
        "is_final": True | False,
        "final_answer": "..."
    }}

    If you have the final answer, set "is_final": true and put the answer in "final_answer".
    """
)

planner_p = PromptTemplate(
    input_variables=["messages"],
    template="""
You are a task breakdown specialist.
Based on this converstion and the last message:
{messages}

Break down the task into a series of steps if needed. If the task is simple and can be handled directly, output "No breakdown needed".

Output in JSON format:
{{
    "breakdown": "NEEDED | NOT_NEEDED",
    "steps": ["step1", "step2", ...]
}}
"""
)