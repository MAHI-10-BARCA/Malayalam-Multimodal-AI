import speech_recognition as sr


def speech_to_text(path):

    r = sr.Recognizer()

    try:
        with sr.AudioFile(path) as source:
            audio = r.record(source)

        return r.recognize_google(audio, language="ml-IN")

    except:
        return ""


def predict_from_speech(path, model, vectorizer):

    from src.text.text_model import predict_text

    text = speech_to_text(path)

    if not text:
        return "uncertain", 0.0

    return predict_text(text, model, vectorizer)