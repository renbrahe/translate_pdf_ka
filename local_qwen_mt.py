# local_qwen_mt.py
import os
from typing import List, Dict, Optional, Callable, Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class QwenLocalTranslator:
    """
    Локальный переводчик на базе Qwen2.5-Instruct (LLM).
    Работает через диалоговый промпт: даём список фрагментов и просим вернуть JSON.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        model_name: имя модели на Hugging Face
        device: "cuda" / "cpu"; по умолчанию auto
        dtype: тип тензоров (fp16 для экономии памяти на GPU; на CPU всё равно будет медленно)
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        print(f"⏳ Загружаем Qwen-модель: {model_name} (device={device})...")

        # токенайзер
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # без bitsandbytes и 4bit, просто обычная модель
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype if device == "cuda" else torch.float32,
        )

        # переводим на нужное устройство
        self.model = self.model.to(device)
        self.model.eval()

        print("✅ Qwen-модель загружена.")

    @staticmethod
    def _build_system_prompt(src_lang: str, tgt_lang: str) -> str:
        return (
            "You are a professional legal and technical translator.\n"
            f"Translate the given text fragments from {src_lang} to {tgt_lang}.\n"
            "Return ONLY a valid JSON object with the following structure:\n"
            "{\n"
            '  "translations": [\n'
            '    {"id": <int>, "text": "<translation text>"},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Preserve all facts, numbers, dates, names and logical structure.\n"
            "- Use natural, formal, high-quality legal language in the target language.\n"
            "- Do NOT add commentary or extra fields to the JSON.\n"
            "- Each item.id must equal the original id from input.\n"
        )

    @staticmethod
    def _build_user_payload(items: List[Tuple[int, str]], src_lang: str, tgt_lang: str) -> str:
        import json
        payload = {
            "source_language": src_lang,
            "target_language": tgt_lang,
            "items": [{"id": i, "text": t} for i, t in items],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _call_model_raw(self, system_prompt: str, user_content: str, max_new_tokens: int = 512) -> str:
        """
        Один прямой вызов LLM (Qwen) в формате chat: system + user → assistant.
        Возвращает сырую строку ответа модели.
        """
        tokenizer = self.tokenizer
        model = self.model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # детерминированно
                num_beams=1,
            )

        gen_ids = output_ids[0, input_ids.shape[-1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def _parse_json_response(raw: str) -> Dict[int, str]:
        """
        Парсит JSON вида:
        {"translations":[{"id":0,"text":"..."}, ...]}
        Возвращает {id: text}.
        """
        import json

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Попытка вырезать JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    raise ValueError(f"Невозможно распарсить JSON из ответа модели:\n{raw}")
            else:
                raise ValueError(f"Ответ модели не похож на JSON:\n{raw}")

        translations = data.get("translations", [])
        result: Dict[int, str] = {}
        if not isinstance(translations, list):
            raise ValueError("Ключ 'translations' должен быть списком объектов {id, text}.")

        for obj in translations:
            if not isinstance(obj, dict):
                continue
            tid = obj.get("id")
            ttext = obj.get("text")
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if isinstance(ttext, str) and ttext.strip():
                result[tid_int] = ttext.strip()

        return result

    def translate(
        self,
        fragments: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 1,
        max_new_tokens: int = 256,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        start: float = 10.0,
        end: float = 90.0,
        chunks_fn=default_chunks,
    ) -> Dict[str, str]:
        """
        Основной метод перевода:
        - dedup фрагментов,
        - режем на батчи и шлём в Qwen,
        - собираем mapping {оригинал -> перевод}.
        """
        if progress_callback is None:
            def progress_callback(p, m):  # type: ignore
                pass

        cleaned = []
        for f in fragments:
            if isinstance(f, str):
                s = f.strip()
                if s:
                    cleaned.append(s)

        if not cleaned:
            return {}

        unique_texts: List[str] = []
        seen = set()
        for s in cleaned:
            if s not in seen:
                seen.add(s)
                unique_texts.append(s)

        id_to_text = {i: t for i, t in enumerate(unique_texts)}
        text_to_id = {t: i for i, t in id_to_text.items()}

        total = len(unique_texts)
        done = 0
        id_to_translated: Dict[int, str] = {}

        system_prompt = self._build_system_prompt(src_lang, tgt_lang)

        for batch_ids in chunks_fn(list(id_to_text.keys()), batch_size):
            items = [(i, id_to_text[i]) for i in batch_ids]
            user_content = self._build_user_payload(items, src_lang, tgt_lang)

            print(f"--> [Qwen {src_lang}->{tgt_lang}] batch, size={len(items)}, first_id={items[0][0]}")
            raw = self._call_model_raw(system_prompt, user_content, max_new_tokens=max_new_tokens)

            try:
                parsed = self._parse_json_response(raw)
            except Exception as e:
                print("⚠️ Ошибка разбора JSON-ответа от Qwen:")
                print(raw)
                raise e

            id_to_translated.update(parsed)

            done += len(batch_ids)
            print(f"   [Qwen {src_lang}->{tgt_lang}] переведено {done}/{total} уникальных фрагментов")

            frac = done / total
            pct = start + (end - start) * frac
            progress_callback(pct, f"Перевод локальной LLM (Qwen {src_lang}->{tgt_lang})...")

        mapping: Dict[str, str] = {}
        for txt, tid in text_to_id.items():
            t = id_to_translated.get(tid)
            if isinstance(t, str) and t.strip():
                mapping[txt] = t
            else:
                mapping[txt] = txt

        return mapping