"""Исключения модуля ``/llm``.

Иерархия ошибок позволяет вызывающему коду обрабатывать конфигурационные
проблемы отдельно от ошибок обращения к API и от общей ошибки модуля.
"""

from __future__ import annotations

import re
from typing import Any

# Маски для секретов, если они случайно попали в текст ошибки:
# ключи вида «sk-...», токены ВКонтакте («vk1.a....») и токены Telegram («123:ABC...»)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{4,}"),
    re.compile(r"\bvk1\.[a-z]\.[A-Za-z0-9_\-]+"),
    re.compile(r"\b\d{5,}:[A-Za-z0-9_\-]{20,}"),
)


class LLMError(Exception):
    """Базовое исключение модуля ``/llm``.

    Все ошибки модуля являются наследниками этого класса, поэтому достаточно
    перехватывать ``LLMError``, чтобы не упустить ошибку клиента.
    """


class LLMConfigError(LLMError):
    """Ошибка конфигурации клиента (например, не задан API-ключ).

    Обычно означает, что обязательная переменная окружения отсутствует
    или содержит пустое значение.
    """


class LLMAPIError(LLMError):
    """Ошибка обращения к API (сеть, таймаут, HTTP-статус, лимит запросов).

    Args:
        message: Человекочитаемое описание ошибки без секретов.
        status_code: HTTP-статус ответа, если он известен.

    Attributes:
        status_code: Код ответа сервера или ``None`` для сетевых ошибок.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code

    def __str__(self) -> str:
        """Возвращает описание ошибки со статусом, если он есть."""
        if self.status_code is not None:
            return f"{super().__str__()} (HTTP {self.status_code})"
        return super().__str__()


def mask_secrets(text: str) -> str:
    """Маскирует известные форматы секретов в произвольном тексте.

    Поддерживаются ключи ProxyAPI (``sk-...``), токены ВКонтакте (``vk1.a....``)
    и токены Telegram (``123:ABC...``). Используется перед записью текста
    ошибки в лог, чтобы секреты не попали в лог-файлы.

    Args:
        text: Исходный текст.

    Returns:
        Текст с заменёнными фрагментами-секретами.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("***", text)
    return text


def describe_exception(exc: BaseException) -> str:
    """Возвращает безопасную для логов строку-описание исключения API.

    Извлекает сообщение из тела ответа (поле ``error.message``), если оно есть,
    иначе возвращает имя типа исключения. Формат ответа и секреты
    в возвращаемую строку не включаются, найденные ключи маскируются.

    Args:
        exc: Перехваченное исключение.

    Returns:
        Однострочное описание ошибки.
    """
    body: Any = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = error["message"]
        elif isinstance(error, str):
            detail = error
    if not detail:
        raw = str(exc).strip()
        detail = raw.splitlines()[0][:300] if raw else type(exc).__name__
    return mask_secrets(detail)[:300]


__all__ = ("LLMError", "LLMConfigError", "LLMAPIError", "describe_exception", "mask_secrets")
