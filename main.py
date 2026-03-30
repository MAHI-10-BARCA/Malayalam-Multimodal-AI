import sys
sys.path.append(".")

# ================================
# 📦 IMPORTS
# ================================
from src.text.text_model import (
    load_labeled_data,
    apply_preprocessing,
    train_model,
    predict_text
)

from src.image.image_pipeline import predict_from_image
from src.image.text_postprocess import normalize_text
from src.visuals.news_renderer import generate_news_image
from src.image.ocr_engine import extract_text
from src.image.image_preprocess import preprocess_image


# ================================
# 🚀 TRAIN MODEL
# ================================
print("🚀 Training model...\n")

df = load_labeled_data()
df = apply_preprocessing(df)

model, vectorizer = train_model(df)


# ================================
# 📝 TEXT TEST
# ================================
print("\n📝 Testing TEXT input...\n")

sample_text = "ഇന്ത്യ ടീം മികച്ച പ്രകടനം കാഴ്ചവെച്ചു"

pred, conf = predict_text(sample_text, model, vectorizer)

print("Text:", sample_text)
print("Prediction:", pred)
print("Confidence:", conf)

if pred != "uncertain":
    output_path = generate_news_image(
        text=sample_text,
        category=pred,
        output_path="text_news.jpg"
    )
    print("🖼️ Text News Image:", output_path)
else:
    print("⚠️ Text too noisy")


# ================================
# 🖼️ IMAGE TEST
# ================================
print("\n🖼️ Testing IMAGE OCR Pipeline...\n")

image_path = "data/image/sample.jpg"

pred, conf = predict_from_image(image_path, model, vectorizer)

print("Prediction:", pred)
print("Confidence:", conf)

if pred != "uncertain":

    processed = preprocess_image(image_path)
    raw_text = extract_text(processed, image_path)
    cleaned_text = normalize_text(raw_text)

    output_path = generate_news_image(
        text=cleaned_text,
        category=pred,
        output_path="image_news.jpg"
    )

    print("🖼️ Image News Image:", output_path)

else:
    print("⚠️ OCR text too noisy")