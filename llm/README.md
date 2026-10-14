# /llm — универсальный клиент нейросети (ProxyAPI)

Модуль для обращения к OpenAI-совместимому API [ProxyAPI](https://proxyapi.ru)
без привязки к Telegram, VK или конкретной продуктовой логике.

## Возможности

- единая точка входа `LLMClient.chat()`;
- конфигурация через переменные окружения, **обязательна только одна** — `PROXYAPI_API_KEY`;
- таймаут и повторные попытки (настраиваются, есть значения по умолчанию);
- извлечение блока размышлений (`reasoning`) «думающих» моделей;
- понятные исключения: `LLMError`, `LLMConfigError`, `LLMAPIError`;
- именованный логгер `llm.client` без вывода секретов.

## Структура

```
llm/
├── __init__.py    # публичный интерфейс
├── client.py      # LLMClient — отправка диалога и разбор ответа
├── config.py      # LLMConfig.from_env() — чтение окружения
├── errors.py      # LLMError / LLMConfigError / LLMAPIError
└── types.py       # ChatMessage, ChatResult, вспомогательные конструкторы
```

## Установка

Зависимости общие для проекта, устанавливаются только в `.venv`:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Конфигурация

| Переменная | Обязательна | По умолчанию | Описание |
|---|---|---|---|
| `PROXYAPI_API_KEY` | **да** | — | ключ ProxyAPI |
| `PROXYAPI_BASE_URL` | нет | `https://api.proxyapi.ru/v1` | адрес API |
| `PROXYAPI_MODEL` | нет | `openai/gpt-6-luna` | модель по умолчанию |
| `PROXYAPI_TIMEOUT` | нет | `90` | таймаут запроса, сек |
| `PROXYAPI_MAX_RETRIES` | нет | `2` | повторные попытки |

## Использование

```python
from llm import LLMClient, ChatMessage, system_message, LLMAPIError

client = LLMClient()          # конфигурация из окружения (.env)

try:
    result = client.chat(
        [
            system_message("Отвечай кратко."),
            ChatMessage("user", "Чем занимаешься?"),
        ],
        model="qwen/qwen3.7-flash",   # необязательно: переопределяет модель
        reasoning_effort="medium",     # необязательно: для Thinking-моделей
    )
    print(result.content)
    print(result.reasoning)            # блок размышлений, если был
except LLMAPIError as exc:
    print("Ошибка API:", exc)

client.close()
```

### Использование в другом проекте

Скопируйте папку `llm/` и положите рядом `.env` с `PROXYAPI_API_KEY=...`
(модуль поднимается по каталогам в поисках `.env`). Либо задайте ключ напрямую:

```python
from llm import LLMClient, LLMConfig

client = LLMClient(LLMConfig(api_key="sk-...", model="openai/gpt-6-luna"))
```

## Логирование

Модуль использует логгер `llm.*`. Настройка уровня и файла — общей функцией
`logging_setup.setup_logging()` из `common/logging_setup.py` (`LOG_LEVEL`, по умолчанию
`INFO`; по умолчанию логи пишутся в консоль **и** в `logs/app.log`). В логи
попадают только модель, число сообщений, размер ответа и время запроса —
ключи и тела запросов не логируются.
