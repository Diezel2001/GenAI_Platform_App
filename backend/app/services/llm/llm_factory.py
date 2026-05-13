from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import json

# =========================
# Base Interface
# =========================

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        pass


# =========================
# OpenAI Wrapper
# =========================

class OpenAIWrapper(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


# =========================
# Ollama (Local LLM)
# =========================

class OllamaWrapper(BaseLLM):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        import requests

        self.model = model
        self.base_url = base_url
        self.requests = requests

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        res = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
        )

        res.raise_for_status()
        return res.json()["response"]


# =========================
# Gemini Wrapper
# =========================

class GeminiWrapper(BaseLLM):
    def __init__(self, model: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.model.generate_content(prompt)
        return response.text


# =========================
# AWS Bedrock Wrapper
# =========================

class BedrockWrapper(BaseLLM):
    def __init__(self, model_id: str):
        import boto3

        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        body = json.dumps({
            "inputText": prompt
        })

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
        )

        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"]


# =========================
# Factory
# =========================

class LLMFactory:
    @staticmethod
    def create(
        provider: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:

        provider = provider.lower()

        if provider == "openai":
            return OpenAIWrapper(
                model=model or "gpt-4o-mini",
                api_key=kwargs.get("api_key"),
            )

        elif provider == "ollama":
            return OllamaWrapper(
                model=model or "llama3",
                base_url=kwargs.get("base_url", "http://localhost:11434"),
            )

        elif provider == "gemini":
            return GeminiWrapper(
                model=model or "gemini-1.5-pro",
                api_key=kwargs.get("api_key"),
            )

        elif provider == "bedrock":
            return BedrockWrapper(
                model_id=model or "amazon.titan-text-express-v1"
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# =========================
# Optional: Config Loader
# =========================

def create_llm_from_env() -> BaseLLM:
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL")

    return LLMFactory.create(
        provider=provider,
        model=model,
    )


# =========================
# Optional: Safe Invoke
# =========================

def safe_invoke(llm: BaseLLM, messages: List[Dict[str, str]]) -> Dict:
    raw = llm.invoke(messages)

    try:
        return json.loads(raw)
    except Exception:
        return {
            "type": "text",
            "content": raw
        }