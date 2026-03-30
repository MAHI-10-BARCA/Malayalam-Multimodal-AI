import pandas as pd


def load_dataset():
    df1 = pd.read_csv("data/text/malayalam.csv")
    df2 = pd.read_csv("data/text/synthetic_data.csv")

    df = pd.concat([df1, df2], ignore_index=True)
    return df


def get_random_news(category):

    df = load_dataset()

    # 🔥 REMOVE BAD ROWS
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    filtered = df[df["label"] == category]

    if filtered.empty:
        return "No news found for this category"

    row = filtered.sample(1).iloc[0]

    # 🔥 FORCE STRING
    return str(row["text"])