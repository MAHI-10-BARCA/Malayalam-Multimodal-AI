import os
import cv2
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont


# ================================
# 📁 GET BACKGROUND
# ================================
def get_background(category):
    mapping = {
        "sports": "sports.jpg",
        "politics": "politics.jpg",
        "business": "business.jpg",
        "entertainment": "entertainment.jpg",
        "world": "world.jpg"
    }

    filename = mapping.get(category, "world.jpg")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    return os.path.join(base_dir, "assets", filename)


# ================================
# 🎯 CONTRAST (CLAHE)
# ================================
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ================================
# 🔪 SHARPEN
# ================================
def sharpen(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)


# ================================
# 🌫️ BLUR BACKGROUND
# ================================
def blur_background(image):
    return cv2.GaussianBlur(image, (15, 15), 0)


# ================================
# 🌈 GRADIENT OVERLAY
# ================================
def add_gradient(image):
    h, w, _ = image.shape
    gradient = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        alpha = i / h
        gradient[i, :] = (0, 0, int(150 * alpha))

    return cv2.addWeighted(image, 0.8, gradient, 0.5, 0)


# ================================
# 🎨 MAIN FUNCTION
# ================================
def generate_news_image(text, category, output_path="generated_news.jpg"):

    # 🔹 Load background
    bg_path = get_background(category)
    image = cv2.imread(bg_path)

    if image is None:
        print("❌ Background load failed:", bg_path)
        return None

    # 🔹 Resize
    image = cv2.resize(image, (800, 600))

    # 🔥 IMAGE PROCESSING PIPELINE
    image = enhance_contrast(image)
    image = sharpen(image)
    image = blur_background(image)
    image = add_gradient(image)

    # 🔹 Convert to PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    # =========================
    # 🔤 FONT LOAD
    # =========================
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    font_path = os.path.join(base_dir, "assets", "fonts", "NotoSansMalayalam-Regular.ttf")

    if not os.path.exists(font_path):
        print("❌ FONT NOT FOUND:", font_path)
        return None

    title_font = ImageFont.truetype(font_path, 42)
    text_font = ImageFont.truetype(font_path, 30)

    # =========================
    # 🔴 HEADER BAR
    # =========================
    draw.rectangle([(0, 0), (800, 90)], fill=(220, 0, 0))

    # Shadow
    draw.text((22, 22), f"{category.upper()} NEWS", font=title_font, fill=(0, 0, 0))
    # Main
    draw.text((20, 20), f"{category.upper()} NEWS", font=title_font, fill=(255, 255, 255))

    # =========================
    # 🌑 BOTTOM OVERLAY
    # =========================
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 140))
    pil_img = Image.alpha_composite(pil_img, overlay)

    draw = ImageDraw.Draw(pil_img)

    # =========================
    # 📝 TEXT WRAP
    # =========================
    text = str(text)
    lines = textwrap.wrap(text, width=25)

    y = 340

    for line in lines:
        # Shadow
        draw.text((42, y + 2), line, font=text_font, fill=(0, 0, 0))
        # Main text
        draw.text((40, y), line, font=text_font, fill=(255, 255, 255))
        y += 42

    # 🔹 Convert back to OpenCV
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

    # 🔹 Save
    success = cv2.imwrite(output_path, final_img)

    if not success:
        print("❌ Failed to save image")
        return None

    print("✅ Premium News Image Generated")

    return output_path