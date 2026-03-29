import cv2
import pytesseract
import re

from src.text.text_model import preprocess_text

# 🔥 Set your Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ================================
# 🖼️ IMAGE PREPROCESSING
# ================================
def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return None

    # Resize for better OCR
    img = cv2.resize(img, None, fx=2, fy=2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blur = cv2.medianBlur(gray, 3)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return thresh


# ================================
# 🔍 OCR FUNCTION
# ================================
def extract_text_from_image(image_path):

    processed = preprocess_image(image_path)

    if processed is None:
        return ""

    # Try multiple configs
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 11"
    ]

    texts = []

    for cfg in configs:
        try:
            text = pytesseract.image_to_string(
                processed,
                lang='mal',
                config=cfg
            )
            texts.append(text)
        except:
            continue

    if not texts:
        return ""

    # Pick longest result (simple heuristic)
    best_text = max(texts, key=len)

    print("\n🔍 OCR TEXT:\n", best_text)

    # Clean text (important)
    clean_text = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', best_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    print("\n🧹 CLEANED TEXT:\n", clean_text)

    return clean_text


# ================================
# 🎯 FINAL PREDICTION PIPELINE
# ================================
def predict_from_image(image_path, model, vectorizer):

    text = extract_text_from_image(image_path)

    if not text or len(text) < 5:
        return "uncertain", 0.0

    clean_text = preprocess_text(text)

    vec = vectorizer.transform([clean_text])

    prediction = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0])

    print("\n🎯 Prediction:", prediction)
    print("Confidence:", confidence)

    return prediction, confidence


# ================================
# 🧪 TEST (OPTIONAL)
# ================================
if __name__ == "__main__":

    from src.text.text_model import load_labeled_data, apply_preprocessing, train_model

    print("🚀 Testing Image Model...\n")

    df = load_labeled_data()
    df = apply_preprocessing(df)

    model, vectorizer = train_model(df)

    pred, conf = predict_from_image("data/image/sample.jpg", model, vectorizer)

    print("\n✅ RESULT:", pred, conf)