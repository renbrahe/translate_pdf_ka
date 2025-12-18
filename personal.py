"""
docx_qwen_qa_offline.py

Простой оффлайн-инструмент:
- Загружает ЛОКАЛЬНУЮ модель Qwen2.5-3B-Instruct из кэша HuggingFace
- НЕ обращается в интернет вообще
- Читает документ .docx
- Даёт задавать вопросы по содержанию документа (на русском)

Зависимости (установить один раз):
    pip install "torch>=2.0" transformers accelerate python-docx

Запуск:
    python docx_qwen_qa_offline.py path/to/document.docx

Команды в интерактивном режиме:
    /exit          – выход
    /reload <path> – загрузить другой DOCX
"""

import os
import sys
import textwrap
from typing import Optional

# ВАЖНО: включаем оффлайн-режим для HF и transformers (чтобы они не лезли в сеть)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ================= НАСТРОЙКИ =================

# Базовая папка с кэшем модели, которую ты указал:
# C:\Users\markov.ai\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct
HF_MODEL_CACHE_DIR = r"C:\Users\markov.ai\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct"

# Чтобы не забивать контекст до отказа — режем текст документа по символам.
MAX_CONTEXT_CHARS = 70000

# Параметры генерации
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0   # 0 = максимально детерминированный ответ


# ================= УТИЛИТЫ =================

def find_local_model_dir(base_dir: str) -> str:
    """
    Находит фактическую папку снапшота с файлами модели внутри кэша HF:
    base_dir/
        snapshots/
            <hash1>/
                config.json, tokenizer_config.json, model.safetensors, ...
            <hash2>/
                ...
    Берёт "последний" снапшот по имени (обычно это самый свежий).
    """
    snapshots_dir = os.path.join(base_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(
            f"Не найдена папка snapshots внутри {base_dir}. "
            f"Убедись, что модель хотя бы один раз была скачана с HuggingFace."
        )

    candidates = [
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"В {snapshots_dir} нет ни одной подпапки со снапшотами модели."
        )

    # Берём последний по алфавиту/имени (обычно работает, т.к. это последние хэши)
    candidates.sort()
    last_snapshot = candidates[-1]
    model_dir = os.path.join(snapshots_dir, last_snapshot)

    # Проверим, что там есть хотя бы config.json
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"В папке снапшота {model_dir} нет config.json. "
            f"Похоже, модель скачана не полностью."
        )

    return model_dir


def read_docx_text(path: str) -> str:
    """
    Читает текст из .docx файла (только абзацы, без таблиц и т.п.).
    При необходимости можно доработать для таблиц/сносок.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    doc = Document(path)
    parts = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)

    full_text = "\n".join(parts)

    if not full_text.strip():
        print("[!] Внимание: документ прочитан, но текст пустой или очень короткий.")
    else:
        print(f"[i] Прочитано символов из DOCX: {len(full_text)}")

    # Ограничиваем размер контекста
    if len(full_text) > MAX_CONTEXT_CHARS:
        print(
            f"[i] Документ длиннее {MAX_CONTEXT_CHARS} символов, "
            f"будет использован только первый фрагмент."
        )
        full_text = full_text[:MAX_CONTEXT_CHARS]

    return full_text


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Загружаем Qwen2.5-3B-Instruct ИСКЛЮЧИТЕЛЬНО из локальной папки кэша.
    Интернет не используется вообще (local_files_only=True + оффлайн-режим).
    """
    # Находим фактическую папку снапшота с моделью
    local_model_dir = find_local_model_dir(HF_MODEL_CACHE_DIR)
    print(f"[i] Локальная папка модели: {local_model_dir}")

    print("[i] Загружаем токенизатор из локальной папки (без интернета)...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
        local_files_only=True,   # <-- ключевой флаг
    )

    # Определяем dtype в зависимости от доступности GPU
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        torch_dtype = torch.float32   # на CPU float16 может не поддерживаться
        device_map = "cpu"

    print("[i] Загружаем модель из локальной папки (без интернета)...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,   # <-- ключевой флаг
    )

    model.eval()
    print("[i] Модель успешно загружена (офлайн).")
    return tokenizer, model


def build_prompt(context: str, question: str) -> str:
    """
    Строим промпт в стиле инструкций.
    Qwen Instruct понимает и чистый текст, и chat template.
    Для простоты используем обычный текстовый prompt.
    """
    prompt = (
        "You are a professional legal and financial analyst.\n"
        "You read documents and answer questions about their content.\n"
        "Use only information from the document; if something is missing, say so.\n\n"
        "=== DOCUMENT START ===\n"
        f"{context}\n"
        "=== DOCUMENT END ===\n\n"
        f"Question (in Russian): {question}\n"
        "Answer in Russian with a clear, structured explanation:\n"
    )
    return prompt


def answer_question(
    tokenizer,
    model,
    context: str,
    question: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Формирует prompt и получает ответ от локальной модели.
    """
    prompt = build_prompt(context, question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # На случай, если модель повторяет prompt: вырезаем кусок после "Answer in Russian"
    marker = "Answer in Russian"
    idx = full_text.rfind(marker)
    if idx != -1:
        answer = full_text[idx + len(marker):].strip(" :\n")
    else:
        # fallback — просто возвращаем всё
        answer = full_text

    return answer.strip()


# ================= ИНТЕРАКТИВНЫЙ РЕЖИМ =================

def interactive_qa(tokenizer, model, initial_doc_path: str):
    """
    Интерфейс:
        - грузим DOCX
        - задаём вопросы в цикле
        - команды:
            /exit          – выход
            /reload <path> – загрузить другой DOCX
    """
    context = None

    def load_doc(path: str) -> Optional[str]:
        try:
            print(f"[i] Загружаю документ: {path}")
            text = read_docx_text(path)
            print("[i] Документ загружен.")
            return text
        except Exception as e:
            print(f"[!] Ошибка при чтении документа: {e}")
            return None

    context = load_doc(initial_doc_path)
    if not context:
        print("[!] Невозможно продолжить без текста документа.")
        return

    print("\n=========================================")
    print("Документ загружен. Теперь можно задавать вопросы.")
    print("Команды:")
    print("  /exit                – выход")
    print("  /reload <path.docx>  – загрузить другой документ")
    print("=========================================\n")

    while True:
        try:
            user_input = input("Вопрос (/exit, /reload path): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[i] Выход.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("[i] Завершение работы.")
            break

        if user_input.startswith("/reload"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("[!] Укажи путь к .docx после /reload")
                continue
            new_path = parts[1].strip('"\' ')
            new_context = load_doc(new_path)
            if new_context:
                context = new_context
            continue

        # Обычный вопрос
        print("\n[i] Модель думает...\n")
        answer = answer_question(tokenizer, model, context, user_input)

        print("====== ОТВЕТ ======")
        print(textwrap.fill(answer, width=100))
        print("===================\n")


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print(f"  python {os.path.basename(__file__)} path/to/document.docx")
        sys.exit(1)

    docx_path = sys.argv[1]

    # Загружаем модель один раз (офлайн)
    tokenizer, model = load_model_and_tokenizer()

    # Запускаем интерактивный режим вопросов/ответов
    interactive_qa(tokenizer, model, docx_path)


if __name__ == "__main__":
    main()
