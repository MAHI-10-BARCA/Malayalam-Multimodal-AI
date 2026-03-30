import numpy as np
import librosa
import soundfile as sf
from scipy.signal import wiener


def reduce_noise(audio_path):
    """
    Applies Wiener filter-based noise reduction.
    """
    print("🔊 Applying noise reduction...")

    try:
        signal, sr = librosa.load(audio_path, sr=16000)

        if np.var(signal) < 1e-6:
            print("⚠️ Low variance signal → skipping")
            return audio_path

        # Wiener filter
        filtered = wiener(signal)

        # Clean NaNs
        filtered = np.nan_to_num(filtered)

        output_path = audio_path.replace(".wav", "_nr.wav")
        sf.write(output_path, filtered, sr)

        return output_path

    except Exception as e:
        print("❌ Noise reduction failed:", e)
        return audio_path