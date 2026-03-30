import pytesseract
import easyocr

# Set path (keep yours)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Lazy load EasyOCR
easyocr_reader = None


def tesseract_ocr(image):
    configs = [
        "--oem 1 --psm 6 -l mal",
        "--oem 1 --psm 4 -l mal",
        "--oem 1 --psm 3 -l mal"
    ]

    texts = []

    for cfg in configs:
        try:
            text = pytesseract.image_to_string(image, config=cfg)
            if text.strip():
                texts.append(text)
        except:
            continue

    if texts:
        return max(texts, key=len)

    return ""


def easyocr_ocr(image_path):
    global easyocr_reader

    if easyocr_reader is None:
        easyocr_reader = easyocr.Reader(['ml'], gpu=False)

    results = easyocr_reader.readtext(image_path)

    text = " ".join([res[1] for res in results])
    return text


def extract_text(image, image_path):
    text = tesseract_ocr(image)

    # Fallback if weak OCR
    if len(text.strip()) < 10:
        text = easyocr_ocr(image_path)

    return text