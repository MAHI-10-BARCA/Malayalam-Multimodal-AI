import cv2
import numpy as np


def remove_noise(image):
    """
    Applies noise reduction using Non-Local Means Denoising.
    """
    print("🔊 Applying Image Denoising...")

    try:
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised

    except Exception as e:
        print("❌ Denoising failed:", e)
        return image