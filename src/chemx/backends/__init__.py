"""Built-in model backends and backend registry."""

from .base import (
    JsonSchema,
    Message,
    ModelBackend,
    ModelError,
    ToolCall,
    ToolDefinition,
)
from .lifecycle import prepare_backend
from .registry import (
    available_backends,
    create_backend,
    get_backend_registration,
    register_backend,
)

# Import built-in implementations so their registration decorators run.
from .deepseek import DeepSeekBackend
from .ollama import OllamaBackend
from .openai import OpenAIBackend

__all__ = [
    "Message",
    "DeepSeekBackend",
    "JsonSchema",
    "ModelBackend",
    "ModelError",
    "OllamaBackend",
    "OpenAIBackend",
    "ToolCall",
    "ToolDefinition",
    "available_backends",
    "create_backend",
    "get_backend_registration",
    "prepare_backend",
    "register_backend",
]
