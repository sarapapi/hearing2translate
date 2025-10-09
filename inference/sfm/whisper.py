from transformers import pipeline

def load_model():
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    return pipe

def generate(model, sample):
    src = sample["src_lang"]
    tgt = sample["tgt_lang"]
    task = "transcribe" if src == tgt else "translate"
    is_long = True # TODO
    try:
        transcriptions = model(sample["sample"], generate_kwargs={"language": src, "task": task}, 
                            # because it doesn't work without
                        chunk_length_s=30, stride_length_s=(5,5) if is_long else (0,0), return_timestamps="word")
    except TypeError:
        transcriptions = model(sample["sample"], generate_kwargs={"language": src, "task": task}, 
                            # because it doesn't work without
                        chunk_length_s=30, stride_length_s=(5,5) if is_long else (0,0), 
                        # mandi/audio/zh/170.wav causes an error if return_timestamps="word" but not with None.
                        # We don't want to change it globally because it would give different outputs,
                        # we would need to re-generate many files :( 
                        return_timestamps=None)
    return transcriptions['text']