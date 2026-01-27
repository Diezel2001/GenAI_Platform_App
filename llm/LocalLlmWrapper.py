import requests
from langchain.llms.base import LLM
from typing import Optional, List
import json


class OllamaLLM(LLM):
    
    model: str = "mistral"
    url: str = "http://localhost:11434"

    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 200
        }

        response = requests.post(
            f"{self.url}/api/generate",
            json=payload
        )
        response.raise_for_status()

        output = ""

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            if "response" in data:
                output += data["response"]

            if data.get("done"):
                break
        # print(output)
        return output


llm = OllamaLLM(
    model="mistral",   # <-- specify model here
)

# Example interaction
# response = llm.invoke("list down 5 fill in the blank quesios where in the blank has the format '<fill>'")


from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain.schema import HumanMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Literal["question", "command"]

graph_builder = StateGraph(State)

def classify_intent(state: State) -> State:
    last_msg = state["messages"][-1].content

    prompt = f"""
    Classify the last message intent into ONE of:
    - question : goal is to obtain information
    - command : goal is to execute a task

    User message:
    "{last_msg}"

    Respond with only the label (question or command).
    """

    response = llm.invoke(prompt).content.strip().lower()

    if response not in {"question", "command"}:
        response = "question"

    return {
        "intent": response
    }

def route_by_intent(state: State):
    return state["intent"]

def handle_question(state: State) -> State:
    answer = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [answer]}

def handle_command(state: State) -> State:
    result = "✅ Command received and executed."
    return {"messages": state["messages"] + [HumanMessage(content=result)]}

# Nodes
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("question", handle_question)
graph_builder.add_node("command", handle_command)

# Edges
graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "question": "question",
        "command": "command",
    }
)

# End all branches
graph_builder.add_edge("question", END)
graph_builder.add_edge("command", END)

graph = graph_builder.compile()

user_input = input("enter a messsage")

state = graph.invoke({"messages": [user_input]})