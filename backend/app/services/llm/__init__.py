"""
LLM Service Module

Provides a factory pattern for creating LLM instances supporting multiple providers:
- OpenAI
- Ollama
- Anthropic
- AWS Bedrock
"""

from backend.app.services.llm.base import BaseLLM, LLMConfig, LLMProvider
from backend.app.services.llm.factory import LLMFactory
from backend.app.services.llm.schemas import (
    OpenAIConfig,
    OllamaConfig,
    AnthropicConfig,
    BedrockConfig,
)

__all__ = [
    "BaseLLM",
    "LLMConfig",
    "LLMProvider",
    "LLMFactory",
    "OpenAIConfig",
    "OllamaConfig",
    "AnthropicConfig",
    "BedrockConfig",
]
