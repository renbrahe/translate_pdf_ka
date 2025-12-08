# quick_test_qwen.py
from local_qwen_mt import QwenLocalTranslator

tr = QwenLocalTranslator(model_name="Qwen/Qwen2.5-36B-Instruct")

fragments = [
    "საქართველოს მარეგულირებელი კომისია ადგენს ტარიფებს.",
    "ეს დოკუმენტი განსაზღვრავს მეთოდოლოგიას.",
]

res = tr.translate(
    fragments,
    src_lang="Georgian",
    tgt_lang="Russian",
    batch_size=1,
    max_new_tokens=128,
)

for k, v in res.items():
    print("----")
    print("GE:", k)
    print("RU:", v)