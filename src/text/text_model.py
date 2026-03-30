import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================================
# 📊 LOAD DATA
# ================================
def load_labeled_data():
    df1 = pd.read_csv("data/text/malayalam.csv")
    df2 = pd.read_csv("data/text/synthetic_data.csv")

    df = pd.concat([df1, df2], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("📊 Dataset size:", df.shape)

    return df


# ================================
# 🧹 PREPROCESS TEXT
# ================================
def preprocess_text(text):
    text = str(text)

    text = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


def apply_preprocessing(df):
    df["clean_text"] = df["text"].apply(preprocess_text)
    print("✅ Preprocessing Done")
    return df


# ================================
# 🧠 TRAIN MODEL
# ================================
def train_model(df):

    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=2,
        max_features=5000
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n🔥 Model Trained")
    print("Accuracy:", acc)

    return model, vectorizer


# ================================
# 🧠 TEXT QUALITY CHECK (NEW 🔥)
# ================================
def is_text_valid(text):
    words = text.split()

    if len(words) < 3:
        return False

    single_chars = sum(1 for w in words if len(w) <= 2)

    if single_chars / len(words) > 0.6:
        return False

    return True


# ================================
# 🎯 PREDICT TEXT (UPDATED 🔥)
# ================================
def predict_text(text, model, vectorizer):

    clean_text = preprocess_text(text)

    # 🚨 Reject garbage text
    if not is_text_valid(clean_text):
        return "uncertain", 0.0

    # 🔥 STRONG RULE SYSTEM
    def keyword_score(text, keywords):
        return sum(1 for k in keywords if k in text)

    scores = {
        "sports": keyword_score(clean_text, ["മത്സരം", "ടീം", "ക്രിക്കറ്റ്"]),
        "politics": keyword_score(clean_text, ["മന്ത്രി", "സർക്കാർ", "പാർട്ടി"]),
        "business": keyword_score(clean_text, ["വിപണി", "ലാഭം", "സാമ്പത്തിക"]),
        "entertainment": keyword_score(clean_text, ["സിനിമ", "ചിത്രം", "നടൻ"]),
        "world": keyword_score(clean_text, ["യുദ്ധം", "രാജ്യം", "അന്താരാഷ്ട്ര"])
    }

    best_category = max(scores, key=scores.get)

    if scores[best_category] >= 2:
        return best_category, 0.95

    # 🚨 Weak text rejection
    if len(clean_text) < 10:
        return "uncertain", 0.0

    vec = vectorizer.transform([clean_text])

    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)
    confidence = max(probs[0])

    # 🚨 Strong confidence filter
    if confidence < 0.75:
        return "uncertain", confidence

    return prediction, confidence