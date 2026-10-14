"""Универсальный клиент нейросети через ProxyAPI (OpenAI-совместимый API).

Публичный интерфейс модуля:

    >>> from llm import LLMClient, ChatMessage
    >>> client = LLMClient()                       # ключ берётся из окружения
    >>> result = client.chat([ChatMessage("user", "Привет!")])
    >>> result.content                              # текст ответа модели

Модуль не зависит от Telegram, VK и продуктовой логики — он готов
к использованию в любом другом проекте.
"""

from __future__ import annotations

from .client import LLMClient
from .config import LLMConfig
from .errors import LLMAPIError, LLMConfigError, LLMError
from .types import (
    ChatMessage,
    ChatResult,
    assistant_message,
    system_message,
    user_message,
)

__all__ = (
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMConfigError",
    "LLMAPIError",
    "ChatMessage",
    "ChatResult",
    "system_message",
    "user_message",
    "assistant_message",
)
