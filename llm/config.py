"""Конфигурация модуля ``/llm``.

Читает переменные окружения (и файл ``.env``) и собирает неизменяемую
конфигурацию клиента :class:`LLMConfig`.

Единственная обязательная переменная — ``PROXYAPI_API_KEY``.
Все остальные параметры имеют значения по умолчанию, поэтому
пользователю не нужно настраивать их вручную.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import LLMConfigError

# ---------------------------------------------------------------------------
# Значения по умолчанию (переопределяются переменными окружения)
# ---------------------------------------------------------------------------

#: Базовый URL ProxyAPI (OpenAI-совместимый API)
DEFAULT_BASE_URL: str = "https://api.proxyapi.ru/v1"
#: Модель по умолчанию
DEFAULT_MODEL: str = "openai/gpt-6-luna"
#: Таймаут одного запроса, секунды
DEFAULT_TIMEOUT: float = 90.0
#: Количество повторных попыток при сетевых ошибках
DEFAULT_MAX_RETRIES: int = 2

# Имена переменных окружения
ENV_API_KEY: str = "PROXYAPI_API_KEY"
ENV_BASE_URL: str = "PROXYAPI_BASE_URL"
ENV_MODEL: str = "PROXYAPI_MODEL"
ENV_TIMEOUT: str = "PROXYAPI_TIMEOUT"
ENV_MAX_RETRIES: str = "PROXYAPI_MAX_RETRIES"

# Плейсхолдеры, которые не считаются настоящим ключом
_KEY_PLACEHOLDERS = ("", "sk-ваш_ключ", "sk-ваш_ключ_здесь", "changeme")


def _load_env_file() -> None:
    """Загружает переменные из ``.env`` в окружение процесса.

    Ищет ``.env`` в каталоге модуля и далее в родительских каталогах (до трёх
    уровней вверх), поэтому модуль работает и в составе этого проекта, и при
    копировании в другой проект со своим ``.env``.

    Примечание:
        Уже заданные переменные окружения имеют приоритет над ``.env``
        (поведение ``load_dotenv`` по умолчанию).
    """
    current = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = current / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            return
        if current.parent == current:
            break
        current = current.parent


def _read_float(name: str, default: float) -> float:
    """Читает вещественное число из переменной окружения.

    Args:
        name: Имя переменной.
        default: Значение по умолчанию.

    Returns:
        Значение переменной или ``default``, если переменной нет либо она
        задана с ошибкой (нечисловое значение).
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    """Читает целое число из переменной окружения.

    Args:
        name: Имя переменной.
        default: Значение по умолчанию.

    Returns:
        Значение переменной или ``default``, если переменной нет либо она
        задана с ошибкой.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    """Неизменяемая конфигурация клиента LLM.

    Attributes:
        api_key: API-ключ ProxyAPI (единственный секретный параметр).
        base_url: Базовый URL API.
        model: Модель по умолчанию.
        timeout: Таймаут одного запроса в секундах.
        max_retries: Количество повторных попыток при временных ошибках.
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Создаёт конфигурацию из переменных окружения (и ``.env``).

        Returns:
            Заполненная конфигурация: обязательна только ``PROXYAPI_API_KEY``,
            остальные параметры берут значения по умолчанию.

        Raises:
            LLMConfigError: если ``PROXYAPI_API_KEY`` отсутствует или содержит
                плейсхолдер вместо реального ключа.
        """
        _load_env_file()

        api_key = os.getenv(ENV_API_KEY, "").strip()
        if api_key in _KEY_PLACEHOLDERS:
            raise LLMConfigError(
                f"Не задана переменная окружения {ENV_API_KEY}. "
                f"Добавьте в файл .env строку: {ENV_API_KEY}=ваш_ключ"
            )

        return cls(
            api_key=api_key,
            base_url=os.getenv(ENV_BASE_URL, "").strip() or DEFAULT_BASE_URL,
            model=os.getenv(ENV_MODEL, "").strip() or DEFAULT_MODEL,
            timeout=_read_float(ENV_TIMEOUT, DEFAULT_TIMEOUT),
            max_retries=_read_int(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
        )


__all__ = (
    "LLMConfig",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_TIMEOUT",
    "ENV_MAX_RETRIES",
)
