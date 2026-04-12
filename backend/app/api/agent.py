
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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


class AgentResponseSchema(BaseModel):
    """Response schema for chat endpoint."""
    
    message: str = Field(..., description="The original message text")
    results: str = Field(..., description="result text")



router = APIRouter()

@router.post("/", response_model=AgentResponseSchema)
async def agent_input_injestion(input: AgentRequestSchema):
    
