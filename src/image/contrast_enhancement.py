import cv2


def enhance_contrast(image):
    """
    Enhances contrast using CLAHE.
    """
    print("🌗 Enhancing contrast using CLAHE...")

    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return enhanced

    except Exception as e:
        print("❌ Contrast enhancement failed:", e)
        return image