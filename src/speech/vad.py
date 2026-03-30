import webrtcvad
import librosa
import numpy as np
import soundfile as sf


def apply_vad(audio_path):
    """
    Applies Voice Activity Detection to remove silence.
    """
    print("🎯 Performing VAD...")

    try:
        signal, sr = librosa.load(audio_path, sr=16000)

        vad = webrtcvad.Vad(1)
        frame_size = int(sr * 0.03)

        speech_frames = []

        for i in range(0, len(signal), frame_size):
            frame = signal[i:i+frame_size]

            if len(frame) < frame_size:
                continue

            frame_bytes = (frame * 32767).astype(np.int16).tobytes()

            if vad.is_speech(frame_bytes, sr):
                speech_frames.append(frame)

        if len(speech_frames) == 0:
            print("⚠️ No speech detected → fallback")
            return audio_path

        speech = np.concatenate(speech_frames)

        output_path = audio_path.replace(".wav", "_vad.wav")
        sf.write(output_path, speech, sr)

        return output_path

    except Exception as e:
        print("❌ VAD failed:", e)
        return audio_path