import google.generativeai as genai
from PIL import Image

# 🔑 PUT YOUR API KEY
genai.configure(api_key="YOUR_API_KEY_HERE")


def extract_text_with_gemini(image_path):

    model = genai.GenerativeModel("gemini-1.5-flash")

    img = Image.open(image_path)

    prompt = """
    Extract only Malayalam text from this image.
    Return clean readable Malayalam sentence.
    """

    response = model.generate_content([prompt, img])

    return response.text.strip()