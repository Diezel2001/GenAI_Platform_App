
from typing import Dict, Type
from pydantic import BaseModel

class ToolDefinition:
    def __init__(self, name: str, schema: Type[BaseModel], func):
        self.name = name
        self.schema = schema
        self.func = func


TOOL_REGISTRY: Dict[str, ToolDefinition] = {}


def register_tool(name: str, schema: Type[BaseModel]):
    def decorator(func):
        TOOL_REGISTRY[name] = ToolDefinition(name, schema, func)
        return func
    return decorator