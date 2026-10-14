"""Типы данных модуля ``/llm``.

Здесь описаны структуры запросов и ответов, чтобы модуль не зависел
от конкретной библиотеки-транспорта в публичном интерфейсе.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

# Допустимые роли сообщений OpenAI-совместимого API
ROLE_SYSTEM: str = "system"
ROLE_USER: str = "user"
ROLE_ASSISTANT: str = "assistant"

MessageDict = Mapping[str, str]
MessageLike = Union["ChatMessage", MessageDict]


@dataclass(frozen=True)
class ChatMessage:
    """Одно сообщение диалога.

    Attributes:
        role: Роль отправителя — ``system``, ``user`` или ``assistant``.
        content: Текст сообщения.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Возвращает словарь формата OpenAI API (``{"role": ..., "content": ...}``)."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatResult:
    """Результат одного обращения к модели.

    Attributes:
        content: Основной текстовый ответ модели.
        reasoning: Блок размышлений (``reasoning``) рассуждающей модели,
            если модель его вернула; иначе ``None``.
        model: Идентификатор модели, которой задавался вопрос.
        usage: Статистика использования токенов (``prompt_tokens`` и т.п.),
            если её вернул сервис; иначе ``None``.
    """

    content: str
    reasoning: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = field(default=None)


def normalize_messages(messages: Sequence[MessageLike]) -> list[dict[str, str]]:
    """Приводит сообщения разных форматов к словарям OpenAI API.

    Принимает как :class:`ChatMessage`, так и обычные словари
    ``{"role": ..., "content": ...}``.

    Args:
        messages: Последовательность сообщений.

    Returns:
        Список словарей ``{"role": str, "content": str}``.

    Raises:
        ValueError: если сообщение имеет неподдерживаемый формат или пустую роль.
    """
    normalized: list[dict[str, str]] = []
    for item in messages:
        if isinstance(item, ChatMessage):
            normalized.append(item.to_dict())
            continue
        if isinstance(item, Mapping):
            role = str(item.get("role", "")).strip()
            content = item.get("content")
            if not role or not isinstance(content, str):
                raise ValueError(
                    "Сообщение должно содержать строковые поля 'role' и 'content'"
                )
            normalized.append({"role": role, "content": content})
            continue
        raise ValueError(f"Неподдерживаемый тип сообщения: {type(item).__name__}")
    return normalized


def system_message(text: str) -> ChatMessage:
    """Создаёт сообщение системного промпта.

    Args:
        text: Текст системной инструкции.

    Returns:
        Сообщение с ролью ``system``.
    """
    return ChatMessage(role=ROLE_SYSTEM, content=text)


def user_message(text: str) -> ChatMessage:
    """Создаёт сообщение пользователя.

    Args:
        text: Текст запроса.

    Returns:
        Сообщение с ролью ``user``.
    """
    return ChatMessage(role=ROLE_USER, content=text)


def assistant_message(text: str) -> ChatMessage:
    """Создаёт сообщение ассистента (для продолжения диалога).

    Args:
        text: Текст предыдущего ответа модели.

    Returns:
        Сообщение с ролью ``assistant``.
    """
    return ChatMessage(role=ROLE_ASSISTANT, content=text)


__all__ = (
    "ChatMessage",
    "ChatResult",
    "MessageLike",
    "MessageDict",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "normalize_messages",
    "system_message",
    "user_message",
    "assistant_message",
)
