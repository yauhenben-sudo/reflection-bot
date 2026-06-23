import os
import json
import asyncio
import random
from datetime import datetime, time, timedelta
from pathlib import Path

import pytz
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI


# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN_REFLECTION"]
OPENROUTER_API_KEY = os.environ["GEMINI_API_KEY"]
CHAT_ID = int(os.environ["CHAT_ID"])

TIMEZONE = pytz.timezone(os.environ.get("TIMEZONE", "Europe/Minsk"))

DATA_DIR = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
SESSIONS_FILE = DATA_DIR / "sessions.json"
STATE_FILE = DATA_DIR / "reflection_state.json"
COMPRESSED_FILE = DATA_DIR / "compressed.json"

SCHEDULER_TICK_SECONDS = 30
MISSED_GRACE_MINUTES = 30
COMPRESS_EVERY_N_SESSIONS = 20

MODEL_NAME = "google/gemini-2.5-flash"
TELEGRAM_MAX_LEN = 4000

QUESTIONS = [
    "Что сегодня дало тебе энергию?",
    "Что сегодня забрало энергию?",
    "В какой момент дня ты чувствовал себя наиболее живым?",
    "Где сегодня мир оказался шире, чем ты ожидал?",
    "Что сегодня оказалось проще, чем казалось заранее?",
    "Было ли сегодня что-то, чего ты раньше избегал, а теперь сделал спокойно?",
    "В какой момент сегодня ты чувствовал себя наиболее свободным?",
    "В какой момент ты чувствовал себя наиболее зажатым обязательствами?",
    "Что сегодня хотелось делать само по себе?",
    "С кем сегодня было приятно общаться?",
    "После какого общения стало легче?",
    "Был ли сегодня человек, рядом с которым ты был собой?",
    "Что сегодня вызвало любопытство?",
    "Какая мысль возвращалась несколько раз?",
    "Что хотелось бы исследовать глубже?",
]

FIXED_SESSION_HOUR = 13   # первая сессия всегда в 13:00
RANDOM_SESSION_START = 18  # вторая сессия — рандомно с 18:00
RANDOM_SESSION_END = 23    # до 23:00


# ─── КЛИЕНТ ───────────────────────────────────────────────────────────────────

client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ─── ФАЙЛЫ ────────────────────────────────────────────────────────────────────

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path, default):
    ensure_data_dir()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            backup = path.with_suffix(".broken.json")
            path.rename(backup)
            print(f"Повреждённый JSON перенесён в {backup}")
    return default


def save_json(path, data):
    ensure_data_dir()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_sessions():
    return load_json(SESSIONS_FILE, [])


def save_sessions(sessions):
    save_json(SESSIONS_FILE, sessions)


def load_state():
    default = {
        "question_queue": [],
        "daily": {
            "date": "",
            "sessions": []
        },
        "pending_session": None,
        "total_sessions": 0
    }
    state = load_json(STATE_FILE, default)
    if "pending_session" not in state:
        state["pending_session"] = None
    if "total_sessions" not in state:
        state["total_sessions"] = 0
    return state


def save_state(state):
    save_json(STATE_FILE, state)


def load_compressed():
    return load_json(COMPRESSED_FILE, {
        "text": "",
        "updated_at": "",
        "sessions_count": 0
    })


def save_compressed(data):
    save_json(COMPRESSED_FILE, data)


# ─── ВРЕМЯ ────────────────────────────────────────────────────────────────────

def now_dt():
    return datetime.now(TIMEZONE)


def today_key():
    return now_dt().strftime("%Y-%m-%d")


def combine_tz(date_str, hour, minute=0):
    naive = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
    return TIMEZONE.localize(naive)


def parse_iso(value):
    return datetime.fromisoformat(value)


# ─── ОЧЕРЕДЬ ВОПРОСОВ ────────────────────────────────────────────────────────

def ensure_question_queue(state):
    if not state.get("question_queue"):
        shuffled = list(range(len(QUESTIONS)))
        random.shuffle(shuffled)
        state["question_queue"] = shuffled
    return state


def pick_two_questions(state):
    state = ensure_question_queue(state)
    queue = state["question_queue"]

    # берём первые два из очереди
    picked_indices = queue[:2]
    state["question_queue"] = queue[2:]

    # если очередь почти кончилась — досыпаем новый цикл
    if len(state["question_queue"]) < 2:
        new_cycle = list(range(len(QUESTIONS)))
        random.shuffle(new_cycle)
        state["question_queue"].extend(new_cycle)

    return [QUESTIONS[i] for i in picked_indices], state


# ─── РАСПИСАНИЕ ──────────────────────────────────────────────────────────────

def generate_daily_sessions(date_str):
    fixed_dt = combine_tz(date_str, FIXED_SESSION_HOUR, 0)

    # рандомное время для второй сессии
    total_minutes = (RANDOM_SESSION_END - RANDOM_SESSION_START) * 60
    offset = random.randint(0, total_minutes)
    random_dt = combine_tz(date_str, RANDOM_SESSION_START, 0) + timedelta(minutes=offset)
    random_dt = random_dt.replace(second=0, microsecond=0)

    return [
        {"id": "s1", "due_at": fixed_dt.isoformat(), "sent": False},
        {"id": "s2", "due_at": random_dt.isoformat(), "sent": False},
    ]


def ensure_today_schedule():
    state = load_state()
    current_date = today_key()

    if state["daily"].get("date") != current_date:
        state["daily"] = {
            "date": current_date,
            "sessions": generate_daily_sessions(current_date)
        }
        save_state(state)

    return state


# ─── КОМПРЕССИЯ ──────────────────────────────────────────────────────────────

async def compress_sessions():
    sessions = load_sessions()

    if len(sessions) < 5:
        return

    sessions_text = "\n\n".join([
        f"Дата: {s['date']}\n"
        f"Вопросы: {s['q1']} / {s['q2']}\n"
        f"Ответ: {s['answer']}"
        for s in sessions
    ])

    prompt = f"""Ты — аналитик поведенческих паттернов. Изучи эти ответы человека на рефлексивные вопросы и составь структурированный аналитический отчёт.

Найди и опиши:
1. Повторяющиеся темы и мотивы (что упоминается чаще всего)
2. Паттерны энергии (что даёт, что забирает, в какое время)
3. Социальные паттерны (с кем хорошо, после кого легче)
4. Паттерны свободы и зажатости
5. Любопытство и интересы (что возвращается, что хочется исследовать)
6. Скрытые закономерности, которые не очевидны на поверхности
7. Динамика — что меняется со временем

Пиши как аналитик, без мотивационного тона. Без markdown-разметки. Конкретно и по делу.

Данные:
{sessions_text}"""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    compressed_text = response.choices[0].message.content.strip()

    save_compressed({
        "text": compressed_text,
        "updated_at": now_dt().isoformat(),
        "sessions_count": len(sessions)
    })

    print(f"Компрессия выполнена: {len(sessions)} сессий")


# ─── ОТПРАВКА СЕССИИ ─────────────────────────────────────────────────────────

async def send_session(bot: Bot, session_id: str):
    state = load_state()
    questions, state = pick_two_questions(state)
    save_state(state)

    msg = f"{questions[0]}\n\n{questions[1]}"

    state["pending_session"] = {
        "session_id": session_id,
        "q1": questions[0],
        "q2": questions[1],
        "asked_at": now_dt().isoformat()
    }
    save_state(state)

    await bot.send_message(chat_id=CHAT_ID, text=msg)

    # Помечаем сессию как отправленную
    state = load_state()
    for s in state["daily"]["sessions"]:
        if s["id"] == session_id:
            s["sent"] = True
    save_state(state)

    print(f"Сессия {session_id} отправлена: {questions[0]} / {questions[1]}")


# ─── ПЛАНИРОВЩИК ─────────────────────────────────────────────────────────────

async def scheduler(bot: Bot):
    while True:
        try:
            ensure_today_schedule()
            state = load_state()
            current = now_dt()

            for session in state["daily"]["sessions"]:
                if session["sent"]:
                    continue

                due_at = parse_iso(session["due_at"])

                if current < due_at:
                    continue

                delay_minutes = (current - due_at).total_seconds() / 60

                if delay_minutes > MISSED_GRACE_MINUTES:
                    # пропустили окно — помечаем как отправленное без отправки
                    session["sent"] = True
                    save_state(state)
                    print(f"Сессия {session['id']} пропущена (опоздание {delay_minutes:.0f} мин)")
                    continue

                await send_session(bot, session["id"])
                break

        except Exception as e:
            print(f"Ошибка планировщика: {e}")

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)


# ─── АНАЛИТИКА ───────────────────────────────────────────────────────────────

async def run_analysis(recent_days=None):
    sessions = load_sessions()
    compressed = load_compressed()

    if not sessions and not compressed["text"]:
        return "Данных пока недостаточно для анализа."

    # фильтрация по периоду если задан
    if recent_days:
        cutoff = now_dt() - timedelta(days=recent_days)
        sessions = [
            s for s in sessions
            if parse_iso(s["date"]) >= cutoff
        ]

    sessions_text = "\n\n".join([
        f"Дата: {s['date']}\nВопросы: {s['q1']} / {s['q2']}\nОтвет: {s['answer']}"
        for s in sessions[-60:]  # не более 60 последних сессий напрямую
    ])

    context_parts = []

    if compressed["text"]:
        context_parts.append(
            f"СЖАТЫЙ АНАЛИЗ ПРЕДЫДУЩИХ ПЕРИОДОВ (по {compressed['sessions_count']} сессиям):\n{compressed['text']}"
        )

    if sessions_text:
        context_parts.append(f"ПОСЛЕДНИЕ СЕССИИ:\n{sessions_text}")

    full_context = "\n\n---\n\n".join(context_parts)

    period_note = f"за последние {recent_days} дней" if recent_days else "за весь период"

    prompt = f"""Проанализируй данные рефлексивных сессий {period_note}.

Найди паттерны, закономерности и выводы, которые не очевидны на поверхности.

Структура анализа:
1. Ключевые паттерны (что повторяется)
2. Энергетические циклы (что даёт и забирает энергию, есть ли ритм)
3. Социальная среда (кто и как влияет)
4. Внутренние тенденции (свобода, зажатость, любопытство)
5. Скрытые связи между темами
6. Динамика — что меняется, что остаётся неизменным
7. Гипотезы для проверки

Без мотивационного тона. Без markdown-разметки. Пиши как аналитик.

{full_context}"""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()


# ─── ОБРАБОТЧИК СООБЩЕНИЙ ────────────────────────────────────────────────────

ANALYZE_TRIGGERS = [
    "проанализируй", "анализ", "статистика", "покажи", "что накопилось",
    "паттерны", "выводы", "итоги", "summary", "analyze"
]


def is_analyze_request(text):
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in ANALYZE_TRIGGERS)


def extract_period(text):
    import re
    normalized = text.lower()
    match = re.search(r"за\s+(\d+)\s*(день|дня|дней|недел|месяц)", normalized)
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        if "недел" in unit:
            return n * 7
        if "месяц" in unit:
            return n * 30
        return n
    if "за неделю" in normalized:
        return 7
    if "за месяц" in normalized:
        return 30
    return None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != CHAT_ID:
        return

    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()

    # запрос аналитики
    if is_analyze_request(text):
        await update.message.reply_text("Анализирую данные, подожди немного...")

        try:
            period = extract_period(text)
            analysis = await run_analysis(recent_days=period)

            # разбиваем на части если длинный
            while len(analysis) > TELEGRAM_MAX_LEN:
                split_at = analysis.rfind("\n", 0, TELEGRAM_MAX_LEN)
                if split_at == -1:
                    split_at = TELEGRAM_MAX_LEN
                await update.message.reply_text(analysis[:split_at].strip())
                analysis = analysis[split_at:].strip()

            await update.message.reply_text(analysis)

        except Exception as e:
            await update.message.reply_text(f"Ошибка при анализе: {e}")
            print(f"Ошибка анализа: {e}")

        return

    # обычный ответ на сессию
    state = load_state()
    pending = state.get("pending_session")

    if not pending:
        await update.message.reply_text(
            "Принял. Следующая сессия придёт по расписанию."
        )
        return

    # сохраняем ответ
    sessions = load_sessions()

    session_record = {
        "date": now_dt().isoformat(),
        "session_id": pending["session_id"],
        "q1": pending["q1"],
        "q2": pending["q2"],
        "answer": text,
        "asked_at": pending["asked_at"],
        "answered_at": now_dt().isoformat()
    }

    sessions.append(session_record)
    save_sessions(sessions)

    # сбрасываем pending
    state["pending_session"] = None
    state["total_sessions"] = state.get("total_sessions", 0) + 1
    save_state(state)

    # компрессия если нужно
    if state["total_sessions"] % COMPRESS_EVERY_N_SESSIONS == 0:
        try:
            await compress_sessions()
        except Exception as e:
            print(f"Ошибка компрессии: {e}")

    await update.message.reply_text("Записал.")


# ─── СТАРТ ────────────────────────────────────────────────────────────────────

async def post_init(application: Application):
    ensure_today_schedule()
    asyncio.create_task(scheduler(application.bot))

    state = load_state()
    current = now_dt()

    print("Бот рефлексии запущен.")
    print(f"Часовой пояс: {TIMEZONE.zone}")
    print(f"Текущее время: {current.strftime('%A %d.%m.%Y %H:%M')}")
    print(f"Дата расписания: {state['daily']['date']}")
    print(f"Сессии сегодня:")
    for s in state["daily"]["sessions"]:
        due = parse_iso(s["due_at"]).strftime("%H:%M")
        print(f"  {s['id']} — {due} — отправлена: {s['sent']}")
    print(f"Всего сессий накоплено: {state.get('total_sessions', 0)}")


def main():
    app = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    app.run_polling()


if __name__ == "__main__":
    main()
