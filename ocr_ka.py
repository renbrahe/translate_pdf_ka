import os
import pytesseract
from PIL import Image

# путь к tesseract.exe (user-install)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\markov.ai\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

img_path = "212 - 0004.jpg"

# формируем имя txt: "212 - 0001.txt"
base, _ = os.path.splitext(img_path)
txt_path = base + ".txt"

# OCR
text = pytesseract.image_to_string(
    Image.open(img_path),
    lang="kat",
    config="--oem 1 --psm 6"
)

# запись в файл
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"OCR сохранён в: {txt_path}")

