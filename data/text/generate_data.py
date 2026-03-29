import pandas as pd
import random

# ---------------- WORD BANK ----------------

subjects = ["ഇന്ത്യ", "കേരളം", "ടീം", "സംഘം", "രാജ്യം"]

sports = [
    "മത്സരം വിജയിച്ചു",
    "മികച്ച പ്രകടനം കാഴ്ചവെച്ചു",
    "കിരീടം നേടി",
    "ടീം വിജയിച്ചു"
]

politics = [
    "മന്ത്രി പ്രഖ്യാപിച്ചു",
    "സർക്കാർ പദ്ധതി അവതരിപ്പിച്ചു",
    "പാർട്ടി യോഗം ചേർന്നു",
    "രാഷ്ട്രീയ പ്രസ്താവന നടത്തി"
]

business = [
    "വിപണി ഉയർന്നു",
    "കമ്പനി ലാഭം നേടി",
    "സാമ്പത്തിക വളർച്ച രേഖപ്പെടുത്തി",
    "വ്യാപാരത്തിൽ പുരോഗതി"
]

entertainment = [
    "സിനിമ വിജയമായി",
    "ചിത്രം ശ്രദ്ധ നേടി",
    "നടന്റെ പ്രകടനം മികച്ചത്",
    "ചിത്രം പ്രേക്ഷകർ ഏറ്റെടുത്തു"
]

world = [
    "യുദ്ധം ആരംഭിച്ചു",
    "സംഘർഷം രൂക്ഷമായി",
    "രാജ്യങ്ങൾ തമ്മിൽ പ്രശ്നം",
    "അന്താരാഷ്ട്ര വാർത്ത"
]


# ---------------- GENERATOR ----------------

def make_sentences(subjects, actions, label, n):
    data = []

    for _ in range(n):
        s = random.choice(subjects)
        a = random.choice(actions)

        # short sentence
        text = f"{s} {a}"
        data.append([text, label])

        # long sentence
        text2 = f"{s} ഇന്നലെ {a} വളരെ ശ്രദ്ധേയമായ സംഭവം ആയിരുന്നു"
        data.append([text2, label])

        # OCR noise (no space)
        text3 = text.replace(" ", "")
        data.append([text3, label])

    return data


def generate_dataset(n=800):

    data = []

    data += make_sentences(subjects, sports, "sports", n)
    data += make_sentences(subjects, politics, "politics", n)
    data += make_sentences(subjects, business, "business", n)
    data += make_sentences(subjects, entertainment, "entertainment", n)
    data += make_sentences(subjects, world, "world", n)

    df = pd.DataFrame(data, columns=["text", "label"])

    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv("data/text/synthetic_data.csv", index=False)

    print("🔥 Dataset generated!")
    print("Total samples:", len(df))


# ---------------- RUN ----------------
if __name__ == "__main__":
    generate_dataset(800)