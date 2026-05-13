from langchain_classic.prompts import PromptTemplate

message_analyzer_p = PromptTemplate(
    input_variables=["message", "convo"],
    template="""
You are a message analyzer.

Analyze user message and determine:
-actual message only if it needed editing or incomplete   
-message type (QUESTION, COMMAND, INSTRUCTION, STATEMENT)                                                                                               
-what is being asked or meaning of the message? what does the user want?
-message complexity (LOW, MEDIUM or HIGH)
-external data dependency - is it explicitly/implicitly dependent on a document.
-tool dependency (NONE, LOW, MEDIUM, HIGH)
-cognitive operations required, atmost 5 (e.g., comparison, synthesis, planning)

Conversation so far:
{convo}

user message: 
{message}
                                                
Analysis should be your reasoning about the message
should be short (one to two paragraph long) compiling  all relevant information.
Output strictly in JSON format:
{{
    "analysis": "..."
}}
"""
)

router_p = PromptTemplate(
    input_variables=["messages", "analysis", "user_message"],
    template="""
You are a routing classifier for an AI system. Your job is to decide how a user message should be handled.

You MUST choose exactly one route:
* "DIRECT" → simple question, can be answered immediately with knowledge only
* "REACT" → requires tool use, external data, actions, or step-by-step reasoning with tools
* "PLAN" → complex, multi-step task that should be broken down into a plan before execution

Return ONLY valid JSON. No explanations.
Output format:
{{
"route": "DIRECT" | "REACT" | "PLAN",
"reason": "<short reason>",
"confidence": 0.0-1.0
}}
---

Decision rules:
1. Use "DIRECT" if:
* The question is factual, explanatory, or conversational
* No tools or external data are required
* Can be answered in one response
Examples:
* "What is Redis?"
* "Explain backpropagation"
* "Summarize this concept"
---

2. Use "REACT" if:
* The task requires tools (search, database, APIs, file access)
* Requires real-time or external data
* Involves taking actions (execute code, retrieve data, etc.)
* May require multiple reasoning steps WITH tools
Examples:
* "Search GitHub for FastAPI repos"
* "Get my latest transactions"
* "Find the weather in Tokyo right now"
---

3. Use "PLAN" if:
* The task is complex and multi-step
* Requires breaking down into a sequence of steps
* Steps depend on each other
* Long-running or structured workflow
Examples:
* "Build and deploy a REST API with Postgres"
* "Create a business plan for a startup"
* "Analyze a dataset and generate insights"
---

Important rules:
* If unsure between "direct" and "react", choose "react"
* If the task can be solved in one step, DO NOT choose "plan"
* Only choose "plan" if decomposition is clearly beneficial
* Be conservative: prefer "direct" over "plan"
---

Given Conversation so far:
{messages}

Given user message:
{user_message}

and Given Analysis:
{analysis}

"""
)

reason_prompt = PromptTemplate(
    input_variables=["tools", "user_message", "observations", "analysis"],

    template="""
You are a reasoning node in a ReAct agent loop.

You are given:
- USER MESSAGE
- ANALYSIS OF USER MESSAGE
- PREVIOUS OBSERVATIONS (if any)

Decide:
- If a tool can answer or help get to the answer of the question/message, use the tool.
- If observation already answers the question/message, give final answer

STRICT RULES:
- do not make up a tool for action, only use available tools given
- If a tool can improve the answer, use it
- Do NOT guess tool outputs
- Do NOT hallucinate tool results
- Output ONLY valid JSON.
- No explanations outside JSON.

AVAILABLE TOOLS:
{tools}

USER MESSAGE:
{user_message}

ANALYSIS OF USER MESSAGE:
{analysis}

PREVIOUS OBSERVATIONS:
{observations}

# OUTPUT FORMAT (STRICT JSON ONLY)
Valid format 1 — Tool call:
{{
    "type": "tool",
    "thought": "brief reasoning for selecting tool",
    "action": "tool_name",
    "args": {{...}}
}}
Valid format 2 — Final answer:
{{
    "type": "final",
    "thought": "brief reasoning for concluding",
    "final_answer": "final response to user"
}}

# TOOL USAGE RULES
- When a tool is used, set:
  - "type": "tool"
  - "action": must match exactly one tool name
  - "args": must match tool parameters exactly
- If no tool is needed, return final answer:
  - "type": "final"
  - "final_answer": concise and correct response

# TOOL SAFETY RULES
- Never invent tool names
- Never omit required arguments
- If unsure about arguments, choose "final" and explain briefly
- Do not assume tool outputs without calling tools

"""
)

observe_prompt = PromptTemplate(
    input_variables=["user_message", "result", "analysis"],

    template="""
You are an observation node.

Your job:
- Interpret tool results
- Extract only useful information
- Keep it concise and factual

USER QUESTION:
{user_message}

ANALYSIS OF USER MESSAGE:
{analysis}

TOOL RESULT:
{result}

Rules:
- Output ONLY valid JSON.
- No explanations outside JSON.

Output STRICT JSON only:
{{
    "obs": "A short, clear summary of what was learned from the tool.",
}}

"""
)

planner_p = PromptTemplate(
    input_variables=["messages"],
    template="""
You are a task planner.

Your job is to break down a user request into clear, executable steps.

RULES:
- Each step must be atomic and actionable
- Steps must be sequential and dependent when necessary
- Avoid vague steps like "analyze" or "think"
- Prefer steps that can be solved with tools or reasoning
- Keep steps minimal but complete

If the task is simple, return NOT_NEEDED.

Conversation:
{messages}

OUTPUT STRICT JSON:
{{
    "breakdown": "NEEDED" | "NOT_NEEDED",
    "steps": ["step 1", "step 2", "..."]
}}
"""
)

aggregator_p = PromptTemplate(
    input_variables=["user_request", "steps_results", "analysis"],
    template="""
You are a final answer synthesizer.

You are given:
1. The original user request
2. analysis of user request
3. A list of steps and their results

Your job:
- Combine the results into a coherent final answer
- Ensure the answer directly satisfies the user request
- Do NOT include step-by-step breakdown unless necessary
- Be concise but complete

USER REQUEST:
{user_request}

ANALYSIS:
{analysis}

STEPS AND RESULTS:
{steps_results}

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include explanations outside JSON
- Do NOT include markdown

OUTPUT FORMAT:
{{
    "final_answer": "your final answer here",
    "confidence": 0.0-1.0
}}
"""
)