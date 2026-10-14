"""Клиент ProxyAPI (OpenAI-совместимый) — универсальная обёртка над LLM API.

Модуль не знает про Telegram, VK и продуктовую логику: он только отправляет
диалог в API и возвращает ответ. Готов к переиспользованию в любом проекте.

Пример:
    >>> from llm import LLMClient, ChatMessage
    >>> client = LLMClient()                      # конфигурация из окружения
    >>> result = client.chat([ChatMessage("user", "Привет!")])
    >>> print(result.content)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .config import LLMConfig
from .errors import LLMAPIError, LLMConfigError, describe_exception
from .types import ChatResult, MessageLike, normalize_messages

logger = logging.getLogger(__name__)

# Ключевые слова в тексте ответа, по которым ищется блок размышлений
_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


class LLMClient:
    """Клиент для обращения к OpenAI-совместимому API (ProxyAPI).

    Потокобезопасен на чтение: один экземпляр можно переиспользовать
    из разных потоков (клиент ``openai`` допускает это).

    Args:
        config: Конфигурация клиента. Если ``None`` — создаётся из
            переменных окружения (:meth:`LLMConfig.from_env`).

    Raises:
        LLMConfigError: если конфигурация не передана и обязательные
            переменные окружения не заданы.

    Attributes:
        config: Используемая конфигурация (без вывода секретов в логи).
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config: LLMConfig = config if config is not None else LLMConfig.from_env()
        self._client: OpenAI | None = None

    # ------------------------------------------------------------------
    # Ленивая инициализация транспорта
    # ------------------------------------------------------------------

    @property
    def client(self) -> OpenAI:
        """Возвращает (и при необходимости создаёт) экземпляр ``openai.OpenAI``."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            # Логируем факт инициализации без ключа и других секретов
            logger.debug(
                "Клиент LLM создан: base_url=%s model=%s timeout=%s",
                self.config.base_url,
                self.config.model,
                self.config.timeout,
            )
        return self._client

    def close(self) -> None:
        """Закрывает HTTP-пул клиента (корректное завершение работы)."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                logger.debug("Клиент LLM закрыт")

    # ------------------------------------------------------------------
    # Основной интерфейс
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[MessageLike],
        *,
        model: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Отправляет диалог в модель и возвращает её ответ.

        Args:
            messages: Сообщения диалога — :class:`ChatMessage` или словари
                ``{"role": ..., "content": ...}``.
            model: Модель для этого запроса. ``None`` — модель из конфигурации.
            temperature: Температура генерации (если поддерживается моделью).
            reasoning_effort: Уровень рассуждений (``minimal`` / ``medium`` /
                ``high``) для «думающих» моделей.
            extra_payload: Дополнительные параметры запроса напрямую в API.

        Returns:
            ChatResult: ответ модели (текст, при наличии — ``reasoning``,
            модель и статистика токенов).

        Raises:
            LLMConfigError: если передан пустой список сообщений.
            LLMAPIError: при ошибке сети, таймауте, лимите запросов или
                ошибке HTTP-статуса.
        """
        payload_messages = normalize_messages(messages)
        if not payload_messages:
            raise LLMConfigError("Список сообщений пуст — нечего отправлять в модель")

        model_name = model or self.config.model
        payload: dict[str, Any] = {"model": model_name, "messages": payload_messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if extra_payload:
            payload.update(extra_payload)

        started = time.monotonic()
        logger.info(
            "Запрос к LLM: model=%s messages=%d", model_name, len(payload_messages)
        )

        try:
            response = self.client.chat.completions.create(**payload)
        except APITimeoutError as exc:
            raise LLMAPIError(
                "Превышено время ожидания ответа от LLM-сервиса"
            ) from exc
        except RateLimitError as exc:
            raise LLMAPIError(
                f"Превышен лимит запросов: {describe_exception(exc)}"
            ) from exc
        except APIConnectionError as exc:
            raise LLMAPIError(
                f"Сетевая ошибка при обращении к {self.config.base_url}: "
                f"{describe_exception(exc)}"
            ) from exc
        except APIStatusError as exc:
            raise LLMAPIError(
                f"Ошибка API: {describe_exception(exc)}",
                status_code=exc.status_code,
            ) from exc

        elapsed = time.monotonic() - started
        result = self._build_result(response, model_name)
        logger.info(
            "Ответ получен: model=%s chars=%d elapsed=%.2fs",
            model_name,
            len(result.content),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Разбор ответа
    # ------------------------------------------------------------------

    @staticmethod
    def extract_reasoning(response: Any) -> str | None:
        """Извлекает блок размышлений рассуждающей модели из ответа API.

        Проверяет несколько типичных мест расположения поля:
        ``choices[0].reasoning_content``, ``choices[0].message.reasoning_content``,
        ``choices[0].message.reasoning``, ``reasoning_text``.

        Args:
            response: Ответ API (объект SDK или словарь).

        Returns:
            Текст размышлений или ``None``, если поле отсутствует.
        """
        try:
            choices: Any = getattr(response, "choices", None)
            if not choices:
                return None
            choice = choices[0]
            containers: list[Any] = [choice]
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            if message is not None:
                containers.append(message)

            for container in containers:
                for field_name in _REASONING_FIELDS:
                    if isinstance(container, dict):
                        value = container.get(field_name)
                    else:
                        value = getattr(container, field_name, None)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except (IndexError, AttributeError, TypeError, KeyError):
            return None
        return None

    @classmethod
    def _build_result(cls, response: Any, model_name: str) -> ChatResult:
        """Собирает :class:`ChatResult` из сырого ответа API.

        Args:
            response: Ответ ``chat.completions.create``.
            model_name: Модель, которой был задан вопрос.

        Returns:
            Результат с текстом ответа и необязательными полями.

        Raises:
            LLMAPIError: если сервис вернул пустой или невалидный ответ.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMAPIError("API вернул ответ без поля choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMAPIError("Модель вернула пустой ответ")

        usage: dict[str, Any] | None = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                "completion_tokens": getattr(raw_usage, "completion_tokens", None),
                "total_tokens": getattr(raw_usage, "total_tokens", None),
            }

        return ChatResult(
            content=content.strip(),
            reasoning=cls.extract_reasoning(response),
            model=getattr(response, "model", None) or model_name,
            usage=usage,
        )


__all__ = ("LLMClient",)
