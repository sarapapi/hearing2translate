from transformers import AutoProcessor, SeamlessM4Tv2Model

def load_model():
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    return model, processor

seamless_lang_codes = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",       
    "it": "ita",
    "nl": "nld",                               
    # TODO
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

    inputs = processor(audios=sample["sample"], return_tensors="pt")

    out = model.generate(**inputs, tgt_lang=tgt, generate_speech=False)[0].cpu().numpy().squeeze()
    print(out)
    return out['text']