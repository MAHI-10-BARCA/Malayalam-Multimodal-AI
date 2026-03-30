from src.image.image_preprocess import preprocess_image
from src.image.ocr_engine import extract_text
from src.image.text_postprocess import normalize_text
from src.text.text_model import preprocess_text, predict_text


def predict_from_image(image_path, model, vectorizer):
    try:
        # ================================
        # 🖼️ PREPROCESS IMAGE
        # ================================
        processed_img = preprocess_image(image_path)

        # ================================
        # 🔍 OCR
        # ================================
        raw_text = extract_text(processed_img, image_path)
        print("\n🔍 OCR TEXT:\n", raw_text)

        # 🚨 Reject weak OCR early
        if not raw_text or len(raw_text.strip()) < 20:
            return "uncertain", 0.0

        # ================================
        # 🧹 CLEAN TEXT
        # ================================
        cleaned_text = normalize_text(raw_text)
        print("\n🧹 CLEANED TEXT:\n", cleaned_text)

        print("\n📊 TEXT LENGTH:", len(cleaned_text))
        print("📊 WORD COUNT:", len(cleaned_text.split()))

        # 🚨 Reject garbage after cleaning
        if not cleaned_text or len(cleaned_text.split()) < 3:
            return "uncertain", 0.0

        # ================================
        # 🎯 FINAL PREDICTION (USES TEXT MODEL)
        # ================================
        prediction, confidence = predict_text(cleaned_text, model, vectorizer)

        print("\n🎯 Prediction:", prediction)
        print("Confidence:", confidence)

        return prediction, confidence

    except Exception as e:
        print("❌ Error:", e)
        return "error", 0.0