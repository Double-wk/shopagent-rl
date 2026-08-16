"""Teacher LLM package: trajectory collection with multiple LLM client backends.

Clients:
- TeacherClient: Anthropic-compatible endpoint (e.g., Zhipu GLM)
- OpenAITeacherClient: OpenAI-compatible endpoint (e.g., GPT-5.6-SOL via mcgrox)
"""
from .client import TeacherClient
from .client_openai import OpenAITeacherClient

__all__ = ["TeacherClient", "OpenAITeacherClient"]
