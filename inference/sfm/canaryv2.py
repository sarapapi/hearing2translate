from nemo.collections.asr.models import ASRModel

def load_model():
    model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
    # downloaded in Dominik's dir:
    #model = ASRModel.restore_from(restore_path="canaryv2/canary-1b-v2/canary-1b-v2.nemo")
    return model


def generate(model, sample):
    src = sample["src_lang"]
    tgt = sample["tgt_lang"]
    audio_path = sample["sample"]
    transcriptions = model.transcribe(audio_path, source_lang=src, target_lang=tgt, timestsamps=True)
    return transcriptions[0].text
