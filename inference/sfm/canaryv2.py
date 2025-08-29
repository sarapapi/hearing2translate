import nemo.collections.asr as nemo_asr

def load_model():
    model = nemo_asr.models.ASRModel.restore_from(restore_path="canaryv2/canary-1b-v2/canary-1b-v2.nemo")
    return model


def generate(model, audio_path, src, tgt):
    transcriptions = model.transcribe(audio_path, source_lang=src, target_lang=tgt)
    return transcriptions[0].text
