import os
import librosa

global DURATION
DURATION = 0.0

def load_model():
    pass

def generate(model, sample):
    audio = sample["sample"]
    if not os.path.exists(audio):
        raise FileNotFoundError(f"Audio file {audio} does not exist.")
    duration = librosa.get_duration(path=audio)
    global DURATION
    DURATION += duration

    return f"{audio} duration: {duration:.2f} seconds"