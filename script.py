"""LangChain-цепочка для генерации кода Telegram-бота по описанию пользователя.

Цепочка состоит из 5 звеньев:

    1. analysis_chain   — анализ пользовательского запроса
    2. build_tools_chain — выделение необходимых инструментов
    3. design_chain      — дизайн и структура кода
    4. code_chain        — генерация полного кода Telegram-бота
    5. review_chain      — проверка сгенерированного кода

Модуль использует локальный LLM-клиент из пакета ``llm`` и модуль LangChain
для структурирования промптов.

Пример запуска::

    python script.py "Бот, который отправляет случайные мемы"
    python script.py "Бот-помощник с командами /start и /help"

Уровень логирования задаётся переменной окружения ``LOG_LEVEL``
(по умолчанию ``INFO``).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from llm import LLMClient, ChatMessage, ChatResult, user_message

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot_generator")

# ---------------------------------------------------------------------------
# Инициализация LLM-клиента (использует ключ из .env)
# ---------------------------------------------------------------------------

llm_client = LLMClient()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def call_llm(messages: list[dict[str, str]]) -> str:
    """Отправляет сообщения через локальный LLM-клиент и возвращает текст ответа.

    Args:
        messages: Список словарей с полями ``role`` и ``content``.

    Returns:
        Текстовый ответ модели.

    Raises:
        LLMAPIError: при ошибке обращения к API или пустом ответе.
    """
    chat_messages = [ChatMessage(m["role"], m["content"]) for m in messages]
    result: ChatResult = llm_client.chat(chat_messages)
    return result.content


def extract_json(raw: str) -> str:
    """Извлекает JSON из текстового ответа LLM.

    Убирает markdown-обёртки `````json ... ````` и лишние текстовые фрагменты.

    Args:
        raw: Исходный текст ответа модели.

    Returns:
        Строка, содержащая валидный JSON-объект.

    Raises:
        ValueError: если из ответа невозможно извлечь JSON.
    """
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        return json_match.group(1)
    # Ищем по скобкам
    brace_count = 0
    end_idx = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    if end_idx > 0:
        return raw[:end_idx]
    # Возвращаем как есть — вызывающий код обработает ошибку
    return raw


# ---------------------------------------------------------------------------
# Chain 1: Анализ пользовательского запроса
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = textwrap.dedent("""\
Ты — опытный аналитик технических заданий для Telegram-ботов.

Пользователь описал бота своими словами. Проанализируй описание и верни
СТРУКТУРИРОВАННЫЙ JSON без лишних текстовых комментариев.

Описание пользователя:
{user_request}

Верни JSON в формате:
{{
    "commands": ["команда1", "команда2"],
    "handlers": ["описание обработчика 1", "описание обработчика 2"],
    "response_format": "формат ответа бота",
    "needs_database": true/false,
    "extra_dependencies": ["aiogram", "другая_библиотека"],
    "description": "краткое описание что делает бот"
}}
""")


def analysis_chain(user_request: str) -> dict[str, Any]:
    """Chain 1: анализирует текстовое описание бота и возвращает структуру задания.

    Извлекает из описания список команд, обработчиков, формат ответа,
    необходимость базы данных и дополнительные зависимости.

    Args:
        user_request: Текстовое описание бота от пользователя.

    Returns:
        Словарь со структурой задания. Ключи:
        ``commands`` (list[str]), ``handlers`` (list[str]),
        ``response_format`` (str), ``needs_database`` (bool),
        ``extra_dependencies`` (list[str]), ``description`` (str).

    Raises:
        ValueError: если модель вернула ответ, из которого не удаётся
            извлечь корректный JSON.
        LLMAPIError: при ошибке обращения к API.
    """
    logger.info("▶ Chain 1: Анализ пользовательского запроса — «%s»", user_request)
    messages = [
        {
            "role": "system",
            "content": "Ты — аналитик технических заданий для Telegram-ботов. Всегда возвращай валидный JSON без текста.",
        },
        {"role": "user", "content": ANALYSIS_PROMPT.format(user_request=user_request)},
    ]
    raw = call_llm(messages)
    logger.info("  Модель: %s | Длина ответа: %d символов", "gpt", len(raw))
    json_str = extract_json(raw)
    analysis: dict[str, Any] = json.loads(json_str)
    logger.info(
        "  Выделено команд: %d | Обработчиков: %d | База данных: %s",
        len(analysis.get("commands", [])),
        len(analysis.get("handlers", [])),
        "да" if analysis.get("needs_database") else "нет",
    )
    logger.info("✓ Chain 1 завершена")
    return analysis


# ---------------------------------------------------------------------------
# Chain 2: Build Tools — выделение инструментов
# ---------------------------------------------------------------------------

TOOLS_PROMPT = textwrap.dedent("""\
Ты — архитектор Telegram-ботов. На основе анализа задания определи,
какие инструменты и библиотеки потребуются для реализации.

Анализ задания:
{analysis_json}

Верни JSON в формате:
{{
    "telegram_api": "aiogram 3.x",
    "middleware": ["описание middleware"],
    "error_handling": "подход к обработке ошибок",
    "logging": "настройки логирования",
    "environment_vars": ["список переменных окружения"],
    "third_party_apis": ["если используются внешние API"]
}}
""")


def build_tools_chain(analysis: dict[str, Any]) -> dict[str, Any]:
    """Chain 2: определяет инструменты и зависимости для реализации бота.

    На основе проанализированного задания определяет, какие API,
    middleware и переменные окружения потребуются.

    Args:
        analysis: Словарь, полученный из ``analysis_chain``.

    Returns:
        Словарь с информацией об инструментах. Ключи:
        ``telegram_api`` (str), ``middleware`` (list[str]),
        ``error_handling`` (str), ``logging`` (str),
        ``environment_vars`` (list[str]), ``third_party_apis`` (list[str]).

    Raises:
        ValueError: если модель вернула ответ, из которого не удаётся
            извлечь корректный JSON.
        LLMAPIError: при ошибке обращения к API.
    """
    logger.info("▶ Chain 2: Определение инструментов и зависимостей")
    messages = [
        {
            "role": "system",
            "content": "Ты — архитектор Telegram-ботов. Всегда возвращай валидный JSON без текста.",
        },
        {
            "role": "user",
            "content": TOOLS_PROMPT.format(
                analysis_json=json.dumps(analysis, ensure_ascii=False)
            ),
        },
    ]
    raw = call_llm(messages)
    logger.info("  Модель: %s | Длина ответа: %d символов", "gpt", len(raw))
    json_str = extract_json(raw)
    tools: dict[str, Any] = json.loads(json_str)
    logger.info(
        "  Telegram API: %s | Переменные окружения: %s",
        tools.get("telegram_api", ""),
        ", ".join(tools.get("environment_vars", [])),
    )
    logger.info("✓ Chain 2 завершена")
    return tools


# ---------------------------------------------------------------------------
# Chain 3: Дизайн и структура кода
# ---------------------------------------------------------------------------

DESIGN_PROMPT = textwrap.dedent("""\
Ты — опытный Python-архитектор Telegram-ботов на aiogram 3.x.

На основе анализа задания и выбранных инструментов спроектируй структуру кода.

Анализ задания:
{analysis_json}

Выбранные инструменты:
{tools_json}

Верни JSON в формате:
{{
    "file_structure": ["список файлов проекта"],
    "main_module": "имя основного файла",
    "handlers_module": "модуль обработчиков",
    "classes": [
        {{"name": "ClassName", "purpose": "описание", "methods": ["метод1", "метод2"]}}
    ],
    "dispatcher_setup": "как настраивается dispatcher",
    "middleware_config": "настройка middleware",
    "startup_sequence": ["пошаговый план запуска"]
}}
""")


def design_chain(analysis: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any]:
    """Chain 3: проектирует архитектуру и структуру файлов бота.

    Определяет модули, классы, настройку dispatcher и middleware.

    Args:
        analysis: Словарь из ``analysis_chain``.
        tools: Словарь из ``build_tools_chain``.

    Returns:
        Словарь с архитектурными решениями. Ключи:
        ``file_structure`` (list[str]), ``main_module`` (str),
        ``handlers_module`` (str), ``classes`` (list[dict]),
        ``dispatcher_setup`` (str), ``middleware_config`` (str),
        ``startup_sequence`` (list[str]).

    Raises:
        ValueError: если модель вернула ответ, из которого не удаётся
            извлечь корректный JSON.
        LLMAPIError: при ошибке обращения к API.
    """
    logger.info("▶ Chain 3: Проектирование структуры кода")
    messages = [
        {
            "role": "system",
            "content": "Ты — Python-архитектор. Всегда возвращай валидный JSON без текста.",
        },
        {
            "role": "user",
            "content": DESIGN_PROMPT.format(
                analysis_json=json.dumps(analysis, ensure_ascii=False),
                tools_json=json.dumps(tools, ensure_ascii=False),
            ),
        },
    ]
    raw = call_llm(messages)
    logger.info("  Модель: %s | Длина ответа: %d символов", "gpt", len(raw))
    json_str = extract_json(raw)
    design: dict[str, Any] = json.loads(json_str)
    logger.info(
        "  Файлов в проекте: %d | Классов: %d",
        len(design.get("file_structure", [])),
        len(design.get("classes", [])),
    )
    logger.info("✓ Chain 3 завершена")
    return design


# ---------------------------------------------------------------------------
# Chain 4: Генерация кода Telegram-бота
# ---------------------------------------------------------------------------

CODE_PROMPT = textwrap.dedent("""\
Ты — опытный Python-разработчик Telegram-ботов на aiogram 3.x.

Сгенерируй ПОЛНЫЙ РАБОЧИЙ код Telegram-бота на основе:

Дизайн проекта:
{design_json}

Анализ задания:
{analysis_json}

Инструменты:
{tools_json}

ТРЕБОВАНИЯ К КОДУ:
1. Использовать aiogram 3.x
2. Все обработчики — async def
3. Токен читается из переменной окружения BOT_TOKEN (или TG_BOT_API_KEY).
   ОБЯЗАТЕЛЬНО загрузить файл .env с помощью python-dotenv
   (from dotenv import load_dotenv; load_dotenv()) ПЕРЕД чтением
   os.getenv — иначе переменные из .env не попадут в окружение и токен не найдётся
4. Никаких заглушек — код должен быть полностью рабочим
5. Правильная настройка dispatcher с middleware
6. Обработка ошибок
7. Грамотная структура с модулями (handlers, filters, etc.)
8. Использование FSM (Finite State Machine) если бот многошаговый
9. Логирование через стандартный logging
10. Корректная обработка shutdown событий

Верни ТОЛЬКО код без markdown-обёрток. Начни сразу с импортов.
""")


def code_chain(
    analysis: dict[str, Any],
    tools: dict[str, Any],
    design: dict[str, Any],
) -> str:
    """Chain 4: генерирует полный рабочий код Telegram-бота на aiogram 3.x.

    На основе спроектированной архитектуры создаёт полный исходный код
    с обработчиками, middleware и корректным запуском.

    Args:
        analysis: Словарь из ``analysis_chain``.
        tools: Словарь из ``build_tools_chain``.
        design: Словарь из ``design_chain``.

    Returns:
        Строка с полным кодом Python-бота.

    Raises:
        ValueError: если модель вернула пустой или некорректный код.
        LLMAPIError: при ошибке обращения к API.
    """
    logger.info("▶ Chain 4: Генерация кода Telegram-бота")
    messages = [
        {
            "role": "system",
            "content": "Ты — Python-разработчик. Верни ТОЛЬКО код Python, без markdown-обёрток и пояснений.",
        },
        {
            "role": "user",
            "content": CODE_PROMPT.format(
                design_json=json.dumps(design, ensure_ascii=False),
                analysis_json=json.dumps(analysis, ensure_ascii=False),
                tools_json=json.dumps(tools, ensure_ascii=False),
            ),
        },
    ]
    raw = call_llm(messages)
    logger.info("  Модель: %s | Длина ответа: %d символов", "gpt", len(raw))
    code = raw.strip()
    # Убираем markdown-обёртки если они есть
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    if not code:
        raise ValueError("Модель вернула пустой код")
    logger.info("  Сгенерировано: %d символов кода", len(code))
    logger.info("✓ Chain 4 завершена")
    return code


# ---------------------------------------------------------------------------
# Chain 5: Review — проверка сгенерированного кода
# ---------------------------------------------------------------------------

REVIEW_PROMPT = textwrap.dedent("""\
Ты — эксперт по кодовым ревью Python-ботов.

Проверь следующий код Telegram-бота и верни JSON с результатами:

Код для проверки:
{code}

Проверь:
1. Синтаксическую корректность
2. Корректность структуры запуска (dispatcher, executor)
3. Корректность импортов aiogram
4. Использование async/await
5. Чтение токена из переменных окружения
6. Обработку ошибок
7. Полноту реализации (нет ли заглушек)

Верни JSON:
{{
    "syntax_ok": true/false,
    "structure_ok": true/false,
    "imports_ok": true/false,
    "async_ok": true/false,
    "env_ok": true/false,
    "errors_handled": true/false,
    "no_stubs": true/false,
    "issues": ["список найденных проблем"],
    "suggestions": ["предложения по улучшению"],
    "overall_score": 0-10
}}
""")


def review_chain(code: str) -> dict[str, Any]:
    """Chain 5: проверяет сгенерированный код на корректность.

    Отправляет код на ревью LLM и возвращает структурированный отчёт
    о качестве кода с оценками по ключевым параметрам.

    Args:
        code: Строка с кодом Python-бота для проверки.

    Returns:
        Словарь с результатами проверки. Ключи:
        ``syntax_ok``, ``structure_ok``, ``imports_ok``, ``async_ok``,
        ``env_ok``, ``errors_handled``, ``no_stubs`` (bool),
        ``issues`` (list[str]), ``suggestions`` (list[str]),
        ``overall_score`` (int 0-10).

    Raises:
        ValueError: если модель вернула ответ, из которого не удаётся
            извлечь корректный JSON.
        LLMAPIError: при ошибке обращения к API.
    """
    logger.info("▶ Chain 5: Проверка сгенерированного кода (%d символов)", len(code))
    messages = [
        {
            "role": "system",
            "content": "Ты — эксперт по кодовым ревью. Всегда возвращай валидный JSON без текста.",
        },
        {"role": "user", "content": REVIEW_PROMPT.format(code=code)},
    ]
    raw = call_llm(messages)
    logger.info("  Модель: %s | Длина ответа: %d символов", "gpt", len(raw))
    json_str = extract_json(raw)
    review: dict[str, Any] = json.loads(json_str)
    logger.info(
        "  Синтаксис (LLM): %s | Структура: %s | Импорты: %s",
        "OK" if review.get("syntax_ok") else "ОШИБКА",
        "OK" if review.get("structure_ok") else "ОШИБКА",
        "OK" if review.get("imports_ok") else "ОШИБКА",
    )
    logger.info("  Оценка: %d/10", review.get("overall_score", 0))
    issues = review.get("issues", [])
    if issues:
        logger.warning("  Проблемы: %s", "; ".join(issues[:3]))
    logger.info("✓ Chain 5 завершена")
    return review


def syntax_check(code: str) -> bool:
    """Минимальная синтаксическая проверка кода через ``ast.parse``.

    Выполняет разбор кода как дерева AST для обнаружения синтаксических
    ошибок, которые LLM-ревью мог пропустить.

    Args:
        code: Строка с кодом Python для проверки.

    Returns:
        ``True`` если синтаксис корректен, ``False`` в противном случае.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as exc:
        logger.warning("  Синтаксическая ошибка: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Основная цепочка — генерация бота
# ---------------------------------------------------------------------------


def generate_bot(user_request: str, output_file: str = "generated_bot.py") -> dict[str, Any]:
    """Запускает полную 5-звенную цепочку генерации Telegram-бота.

    Последовательно выполняет анализ запроса, определение инструментов,
    проектирование архитектуры, генерацию кода и ревью. Результат
    сохраняется в указанный файл.

    Args:
        user_request: Текстовое описание бота от пользователя.
        output_file: Имя файла для сохранения сгенерированного кода.

    Returns:
        Словарь с результатами всех звеньев цепочки. Ключи:
        ``analysis``, ``tools``, ``design``, ``code``, ``review``,
        ``output_file``, ``syntax_ast_ok``, ``all_checks_passed``.

    Raises:
        LLMAPIError: при ошибке обращения к API на любом этапе.
        ValueError: если модель вернула некорректный ответ.
    """
    logger.info("=" * 60)
    logger.info("ГЕНЕРАТОР TELEGRAM-БОТОВ (LangChain Chain)")
    logger.info("=" * 60)
    logger.info("Задание пользователя: «%s»", user_request)
    logger.info("Уровень логирования: %s", LOG_LEVEL)
    logger.info("Файл вывода: %s", output_file)

    # --- Chain 1: Анализ ---
    analysis = analysis_chain(user_request)

    # --- Chain 2: Build Tools ---
    tools = build_tools_chain(analysis)

    # --- Chain 3: Дизайн ---
    design = design_chain(analysis, tools)

    # --- Chain 4: Генерация кода ---
    code = code_chain(analysis, tools, design)

    # --- Chain 5: Review ---
    review = review_chain(code)
    syntax_ok = syntax_check(code)
    review["syntax_ast_ok"] = syntax_ok
    review["all_checks_passed"] = (
        syntax_ok
        and review.get("syntax_ok", False)
        and review.get("structure_ok", False)
        and review.get("imports_ok", False)
    )

    # --- Запись файла ---
    Path(output_file).write_text(code, encoding="utf-8")
    lines = code.count("\n") + 1
    logger.info("Код сохранён в файл: %s (%d строк, %d байт)", output_file, lines, len(code.encode("utf-8")))

    # --- Пробная проверка импорта ---
    logger.info("Проверка сгенерированного файла через subprocess...")
    try:
        check_code = (
            f"import ast; "
            f"ast.parse(open(r'{output_file}', encoding='utf-8').read())"
        )
        result = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("✓ Сгенерированный файл имеет корректный синтаксис Python")
        else:
            logger.warning("Предупреждение при проверке: %s", result.stderr[:200])
    except subprocess.TimeoutExpired:
        logger.warning("Превышено время проверки синтаксиса")
    except Exception as exc:
        logger.warning("Не удалось выполнить проверку: %s", exc)

    logger.info("=" * 60)
    logger.info("ГОТОВО! Все 5 звеньев цепочки выполнены.")
    logger.info("=" * 60)

    return {
        "analysis": analysis,
        "tools": tools,
        "design": design,
        "code": code,
        "review": review,
        "output_file": output_file,
    }


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ГЕНЕРАТОР TELEGRAM-БОТОВ (LangChain Chain)")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("Использование:")
        print('  python script.py "Описание бота"')
        print()
        print("Примеры:")
        print('  python script.py "Бот, который отправляет случайные мемы"')
        print('  python script.py "Бот-помощник с командами /start и /help"')
        print('  python script.py "Бот для опросов с FSM-сценариями"')
        sys.exit(1)

    user_request = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "generated_bot.py"

    results = generate_bot(user_request, output_file)

    print()
    print("─" * 60)
    print("ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ REVIEW")
    print("─" * 60)
    print(json.dumps(results["review"], indent=2, ensure_ascii=False))