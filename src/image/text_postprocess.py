import re


def clean_malayalam_text(text):
    # Keep only Malayalam chars + space
    text = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def fix_spacing(text):
    # Remove weird internal spacing
    text = re.sub(r'(?<=\w)\s+(?=\w)', '', text)

    return text


def normalize_text(text):
    text = clean_malayalam_text(text)
    text = fix_spacing(text)
    return text