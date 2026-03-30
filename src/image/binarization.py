import cv2


def adaptive_binarization(image):
    """
    Converts image to binary using adaptive thresholding.
    """
    print("⚫ Applying adaptive binarization...")

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )

        return thresh

    except Exception as e:
        print("❌ Binarization failed:", e)
        return image