import numpy as np
import sounddevice as sd
import tempfile
from scipy.io.wavfile import write


def record_audio(duration=5, fs=16000):
    """
    Records audio from microphone and saves as WAV file.
    """
    print("🎤 Recording audio...")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Convert to int16
    audio = (audio * 32767).astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, audio)

    print("✅ Audio recorded:", temp_file.name)
    return temp_file.name