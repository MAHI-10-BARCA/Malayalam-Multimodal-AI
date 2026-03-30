import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extracts MFCC features from audio.
    """
    print("🎼 Extracting MFCC features...")

    try:
        signal, sr = librosa.load(audio_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=13
        )

        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        print("❌ Feature extraction failed:", e)
        return np.zeros(13)