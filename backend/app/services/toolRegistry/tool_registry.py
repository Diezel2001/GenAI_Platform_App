from typing import Dict, Type, Callable, Optional
from pydantic import BaseModel


class ToolDefinition:
    def __init__(
        self,
        name: str,
        schema: Type[BaseModel],
        func: Callable
    ):
        self.name = name
        self.schema = schema
        self.func = func

    def __repr__(self):
        return f"ToolDefinition(name={self.name})"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register_tool(self, name: str, schema: Type[BaseModel]):
        """
        Decorator for registering tools.
        """

        def decorator(func: Callable):
            self._tools[name] = ToolDefinition(
                name=name,
                schema=schema,
                func=func
            )
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Retrieve a tool by name.
        """
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, ToolDefinition]:
        """
        Return all registered tools.
        """
        return self._tools


# --------------------------------------------------
# Example Usage
# --------------------------------------------------

# registry = ToolRegistry()


# class AddSchema(BaseModel):
#     a: int
#     b: int


# @registry.register_tool(name="add_numbers", schema=AddSchema)
# def add_numbers(a: int, b: int):
#     return a + b


# tool = registry.get_tool("add_numbers")

# print(tool)
# print(tool.func(2, 3))