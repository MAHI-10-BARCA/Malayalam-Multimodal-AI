import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================================
# 📊 LOAD DATA (MERGE BOTH)
# ================================
def load_labeled_data():

    df1 = pd.read_csv("data/text/malayalam.csv")
    df2 = pd.read_csv("data/text/synthetic_data.csv")

    df = pd.concat([df1, df2], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    print("📊 Dataset size:", df.shape)

    return df


# ================================
# 🧹 PREPROCESS (IMPORTANT)
# ================================
def preprocess_text(text):
    text = str(text)

    # Keep Malayalam only
    text = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', text)

    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


def apply_preprocessing(df):
    df["clean_text"] = df["text"].apply(preprocess_text)

    print("✅ Preprocessing Done")

    return df


# ================================
# 🧠 TRAIN (ROBUST)
# ================================
def train_model(df):

    vectorizer = TfidfVectorizer(
        analyzer='char',        # 🔥 KEY
        ngram_range=(3, 5),     # 🔥 IMPORTANT
        min_df=2
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
# 🎯 PREDICT (HYBRID)
# ================================
def predict_text(text, model, vectorizer):

    clean_text = preprocess_text(text)

    # 🔥 RULE BOOST (helps a lot)
    if any(word in clean_text for word in ["മത്സരം", "ടീം", "ക്രിക്കറ്റ്"]):
        return "sports", 0.95

    if any(word in clean_text for word in ["മന്ത്രി", "സർക്കാർ", "പാർട്ടി"]):
        return "politics", 0.95

    if any(word in clean_text for word in ["വിപണി", "ലാഭം", "സാമ്പത്തിക"]):
        return "business", 0.95

    if any(word in clean_text for word in ["സിനിമ", "ചിത്രം", "നടൻ"]):
        return "entertainment", 0.95

    if any(word in clean_text for word in ["യുദ്ധം", "രാജ്യം", "അന്താരാഷ്ട്ര"]):
        return "world", 0.95

    # 🔥 Filter garbage
    if len(clean_text) < 10:
        return "uncertain", 0.0

    vec = vectorizer.transform([clean_text])

    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)
    confidence = max(probs[0])

    if confidence < 0.6:
        return "uncertain", confidence

    return prediction, confidence


# ================================
# 🧪 TEST
# ================================
if __name__ == "__main__":

    df = load_labeled_data()
    df = apply_preprocessing(df)

    model, vectorizer = train_model(df)

    sample = "ഇന്ത്യ ടീം മികച്ച പ്രകടനം കാഴ്ചവെച്ചു"

    pred, conf = predict_text(sample, model, vectorizer)

    print("\nPrediction:", pred)
    print("Confidence:", conf)