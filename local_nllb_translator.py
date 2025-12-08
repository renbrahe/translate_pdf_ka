# local_nllb_translator.py

from typing import List, Dict, Optional, Callable, Iterable
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# простой генератор батчей – можно использовать твой chunks
def chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def translate_with_nllb_ka_en(
    fragments: List[str],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    start: float = 10.0,
    end: float = 90.0,
) -> Dict[str, str]:
    """
    Локальный перевод грузинский → английский с упором на качество
    через facebook/nllb-200-distilled-600M.

    Требуется:
      pip install transformers torch sentencepiece

    ВНИМАНИЕ: модель тяжёлая, на CPU i3 будет очень медленно.
    """

    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    SRC_LANG = "kat_Geor"   # грузинский
    TGT_LANG = "eng_Latn"   # английский

    print(f"⏳ Загружаем локальную модель NLLB ({MODEL_NAME}) для ka→en...")
    if progress_callback:
        progress_callback(start, "Загрузка локальной модели (NLLB, ka→en)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    device = torch.device("cpu")  # на твоём ноуте лучше явно CPU
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # чистим входные фрагменты
    remaining = [f.strip() for f in fragments if isinstance(f, str) and f.strip()]
    total = len(remaining)
    if total == 0:
        return {}

    mapping: Dict[str, str] = {}

    # Параметры – поджимай, если модель тормозит
    BATCH_SIZE = 2      # для i3 советую начать с 1–2
    MAX_TOKENS = 256    # ограничивает длину вывода

    done = 0

    for i, batch in enumerate(chunks(remaining, BATCH_SIZE), start=1):
        print(f"--> [NLLB ka→en] Начинаю batch {i}, размер {len(batch)}")

        # токенизация
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,     # входной предел (можно уменьшить до 256)
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # генерация
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
                max_length=MAX_TOKENS,
            )

        # декодирование
        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for orig, trans in zip(batch, outputs):
            # если вдруг пусто – подставляем оригинал
            mapping[orig] = trans.strip() if trans and trans.strip() else orig

        done += len(batch)
        print(f"   [NLLB ka→en] Переведено {done}/{total} фрагментов")

        if progress_callback and total > 0:
            frac = done / total
            pct = start + (end - start) * frac
            progress_callback(pct, "Перевод локальной моделью NLLB (ka→en)...")

    return mapping