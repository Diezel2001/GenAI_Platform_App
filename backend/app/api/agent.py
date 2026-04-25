
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.core.workflows import _make_human_message

class AgentRequestSchema(BaseModel):
    """Request schema for chat endpoint."""
    
    message: str = Field(
        ...,
        description="The message text",
        min_length=1,
        max_length=5000,
        examples=["What is the capital of france?"]
    )
    k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    session_id: str


class AgentResponseSchema(BaseModel):
    """Response schema for chat endpoint."""
    
    message: str = Field(..., description="The original message text")
    results: str = Field(..., description="result text")



router = APIRouter()

@router.post("/", response_model=AgentResponseSchema)
async def agent_input_injestion(request: Request, input: AgentRequestSchema):
    # Create and compile the workflow
    graph = request.app.state.graph
    
    # Create human message from request
    human_message = _make_human_message(input.message)
    
    # Invoke the workflow with initial state
    result = graph.invoke(
        {"messages": [human_message]}, 
        config={"configurable": {"thread_id": input.session_id}})
    
    # Extract the last AI message from the response
    messages = result.get("messages", [])
    response_text = ""
    if messages:
        last_message = messages[-1]
        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # Return the response schema
    return AgentResponseSchema(
        message=input.message,
        results=response_text
    )
