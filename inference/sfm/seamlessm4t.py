from transformers import AutoProcessor, SeamlessM4Tv2Model
import librosa

def load_model():
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    return model, processor

seamless_lang_codes = {
    "en": "eng",
    "es": "spa",       
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "zh": "cmn",
    "zh-cn": "cmn",
}

def generate(model, sample):
    model, processor = model
    src = seamless_lang_codes[sample["src_lang"]]
    tgt = seamless_lang_codes[sample["tgt_lang"]]

    task = "transcribe" if src == tgt else "translate"
    is_long = True # TODO

    # this is probably wrong, needs to be worked on next time
    # docs: 
    # https://huggingface.co/facebook/seamless-m4t-v2-large
    # https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2 

    audio, sr = librosa.load(sample["sample"], sr=processor.feature_extractor.sampling_rate)
    inputs = processor(audios=audio, return_tensors="pt")

    out = model.generate(**inputs, tgt_lang=tgt, generate_speech=False)[0].cpu().numpy().squeeze()
    print(out)

    text = processor.decode(out, skip_special_tokens=True)
    return text