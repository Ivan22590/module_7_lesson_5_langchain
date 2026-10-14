# LangChain Telegram Bot Generator

Генератор кода Telegram-ботов на основе LLM. Принимает текстовое описание бота и возвращает полностью рабочий код на **aiogram 3.x**.

## Как это работает

Вы описываете бота обычным текстом — цепочка из 5 звеньев анализирует задание, проектирует архитектуру, генерирует код и проверяет его качество. На выходе — готовый `.py` файл, который можно запустить.

Каждое звено — это **LangChain-цепочка** ``prompt | llm | parser``:
- ``ChatPromptTemplate`` формирует системный и пользовательский промпты
- ``ChatOpenAI`` (через ProxyAPI) вызывает LLM
- ``StrOutputParser`` извлекает сырой текст ответа

```bash
python script.py "Бот, который отправляет случайные мемы"
# → generated_bot.py
```

## Цепочка генерации (5 звеньев)

| # | Звено | Что делает | Результат |
|---|-------|------------|-----------|
| 1 | `analysis_chain` | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` — анализ описания бота | JSON: команды, обработчики, формат ответа, нужна ли БД, зависимости |
| 2 | `tools_chain` | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` — определение инструментов | JSON: API, middleware, обработка ошибок, переменные окружения |
| 3 | `design_chain` | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` — проектирование архитектуры | JSON: структура файлов, классы, настройка dispatcher |
| 4 | `code_chain` | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` — генерация кода | Полный исходник бота на aiogram 3.x |
| 5 | `review_chain` | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` — проверка кода | JSON: оценки по 7 параметрам, проблемы, оценка 0–10 + проверка `ast.parse` |

Дополнительно к LLM-ревью код проверяется синтаксически (`ast.parse`) и через subprocess-тест — это гарантирует отсутствие синтаксических ошибок.

## Установка

```bash
# 1. Создание виртуального окружения
python -m venv .venv

# 2. Активация
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 3. Установка зависимостей
pip install -r requirements.txt
```

## Настройка

1. Скопируйте шаблон переменных окружения:

   ```bash
   copy .env.example .env       # Windows
   cp .env.example .env         # Linux / macOS
   ```

2. Заполните **обязательные** ключи в файле `.env`:

   | Переменная | Обязательная | Описание |
   |------------|:---:|----------|
   | `PROXYAPI_API_KEY` | ✅ | Ключ ProxyAPI для обращений к LLM — [proxyapi.ru](https://proxyapi.ru) |
   | `BOT_TOKEN` | ✅ | Токен Telegram-бота — получить у [@BotFather](https://t.me/BotFather) |
   | `MEME_API_URL` | — | URL API мемов для сгенерированных мем-ботов |
   | `MEME_MEDIA_DIR` | — | Локальная папка с мемами (по умолчанию `memes`) |
   | `LOG_LEVEL` | — | Уровень логирования: `DEBUG` / `INFO` (по умолчанию) / `WARNING` / `ERROR` |
   | `PROXYAPI_MODEL` | — | Модель LLM (по умолчанию `openai/gpt-6-luna`) |
   | `PROXYAPI_BASE_URL` | — | Базовый URL API (по умолчанию `https://api.proxyapi.ru/v1`) |

> ⚠️ Файл `.env` содержит секреты и добавлен в `.gitignore` — не коммитьте его в репозиторий.

## Использование

### Базовый запуск

```bash
python script.py "Бот, который отправляет случайные мемы"
```

Результат сохраняется в `generated_bot.py`.

### С именем выходного файла

```bash
python script.py "Бот-помощник с командами /start и /help" my_bot.py
```

### Примеры описаний

```bash
python script.py "Бот, который отправляет случайные мемы"
python script.py "Бот-помощник с командами /start и /help"
python script.py "Бот для опросов с FSM-сценариями"
python script.py "Бот для отслеживания курса валют"
python script.py "Бот-заметки: добавление, просмотр и удаление записей"
```

### Запуск сгенерированного бота

```bash
# 1. Убедитесь, что BOT_TOKEN заполнен в .env
# 2. Запустите бота
python generated_bot.py
```

Сгенерированный бот сам загружает `.env` через `python-dotenv`, поэтому токен подхватывается автоматически.

## Логирование

Каждое звено цепочки логирует начало и конец работы, модель, объём ответа и результаты проверок:

```
2026-10-14 12:06:43 | INFO     | ▶ Chain 1: Анализ пользовательского запроса — «Бот, который отправляет случайные мемы»
2026-10-14 12:06:49 | INFO     |   Модель: gpt | Длина ответа: 445 символов
2026-10-14 12:06:49 | INFO     |   Выделено команд: 2 | Обработчиков: 2 | База данных: нет
2026-10-14 12:06:49 | INFO     | ✓ Chain 1 завершена
```

Уровень задаётся переменной окружения `LOG_LEVEL`:

| Уровень | Что видно |
|---------|-----------|
| `DEBUG` | Максимально подробно, включая служебные сообщения HTTP |
| `INFO` | Статус каждого звена (по умолчанию) |
| `WARNING` | Только предупреждения (например, проблемы из ревью) |
| `ERROR` | Только критические ошибки |

```powershell
$env:LOG_LEVEL="DEBUG"; python script.py "Бот для опросов"   # Windows
LOG_LEVEL=DEBUG python script.py "Бот для опросов"           # Linux / macOS
```

## Требования к сгенерированному коду

Генератор требует от модели соблюдать стандарт:

- **aiogram 3.x** — `Bot`, `Dispatcher`, `Router`
- Все обработчики — `async def`
- Токен читается из `BOT_TOKEN` / `TG_BOT_API_KEY` **после** загрузки `.env` через `python-dotenv`
- Никаких заглушек — код полностью рабочий
- Настроенный `dispatcher` с middleware и роутерами
- Обработка ошибок (`@router.error()`, `try/except` в обработчиках)
- Логирование через стандартный `logging`
- Корректное завершение работы (закрытие сессий, `shutdown`)
- FSM при многошаговых сценариях

## Структура проекта

```
.
├── script.py              # 5-звенная цепочка генерации (точка входа)
├── llm/                   # Универсальный LLM-клиент (ProxyAPI)
│   ├── __init__.py
│   ├── client.py          # Обёртка над OpenAI-совместимым API
│   ├── config.py          # Конфигурация из переменных окружения
│   ├── errors.py          # Иерархия исключений
│   └── types.py           # Типы сообщений и результатов
├── .env                   # Секреты (в .gitignore)
├── .env.example           # Шаблон переменных окружения
├── .gitignore
├── requirements.txt       # Зависимости
└── README.md
```

Сгенерированные боты (`generated_bot.py` и др.) не версионируются — их всегда можно создать заново.

## Схема работы

```
Пользовательский запрос
        │
        ▼
┌──────────────────────┐
│ analysis_chain        │  ChatPromptTemplate | ChatOpenAI | StrOutputParser
│ (анализ)              │  → команды, обработчики, формат ответа, БД
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ tools_chain           │  ChatPromptTemplate | ChatOpenAI | StrOutputParser
│ (инструменты)         │  → API, middleware, переменные окружения
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ design_chain          │  ChatPromptTemplate | ChatOpenAI | StrOutputParser
│ (дизайн)              │  → структура файлов, классы, dispatcher
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ code_chain            │  ChatPromptTemplate | ChatOpenAI | StrOutputParser
│ (код)                 │  → полный код бота (aiogram 3.x)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ review_chain          │  ChatPromptTemplate | ChatOpenAI | StrOutputParser
│ (ревью)               │  → 7 проверок + оценка 0–10 + ast.parse
└─────────┬────────────┘
          ▼
   generated_bot.py  →  python generated_bot.py
```

## Возможные проблемы

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: No module named 'dotenv'` | Установите зависимости: `pip install -r requirements.txt` |
| `LLMConfigError: Не задана PROXYAPI_API_KEY` | Заполните ключ в `.env` (скопируйте из `.env.example`) |
| Бот не находит токен | Проверьте `BOT_TOKEN` в `.env`; сгенерированный бот загружает `.env` автоматически |
| Низкая оценка в ревью (< 7) | Уточните описание бота и запустите генерацию заново |
| `bot could not find credential` | Токен невалиден — перевыпустите его у @BotFather |

## Зависимости

См. `requirements.txt`:

- `langchain>=0.3.0`, `langchain-openai>=0.3.0`, `langchain-core>=0.3.0` — фреймворк цепочек (`ChatPromptTemplate`, `RunnableSequence`)
- `openai>=1.0.0` — клиент OpenAI-совместимого API (ProxyAPI)
- `python-dotenv>=1.0.0` — загрузка переменных из `.env`
- `aiogram>=3.7,<4.0` — фреймворк Telegram-ботов (для запуска сгенерированных ботов)

Транзитивные зависимости (`httpx`, `aiohttp`, `pydantic` и др.) устанавливаются автоматически через `pip install -r requirements.txt`.

## Лицензия

MIT
