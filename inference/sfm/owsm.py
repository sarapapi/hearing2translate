from transformers import pipeline
#Requirements are
#librosa
#torch
#espnet
#espnet_model_zoo

import soundfile as sf
import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import librosa
from langcodes import Language

import logging

def load_model():
    logging.disable(logging.CRITICAL)
    model = Speech2TextGreedySearch.from_pretrained(
        "espnet/owsm_ctc_v4_1B",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        #dtype="bfloat16",
        use_flash_attn=False,
        generate_interctc_outputs=False,
    )
    return model

def generate(model, sample: dict):

    context = sample.get("context", "long") 

    context_len_in_secs = 4   # left and right context when doing buffered inference, default parameter in HF repo
    batch_size = 1   # Ultra fast, doesnt really matter alot

    src = Language.get(sample["src_lang"]).to_alpha3()
    tgt = Language.get(sample["tgt_lang"]).to_alpha3()
    model.lang_sym=f'<{src}>'
    model.task_sym = "<asr>" if src == tgt else f"<st_{tgt}>"

    #speech, rate = sf.read(sample["sample"]
    #)
    speech, rate = librosa.load(sample["sample"], sr=16000)
    long_form_add_threshold = 0.2
    #We distinguish short inputs and outputs. This is because when doing
    #the long inference decoding on <30 inputs, I observed decreased eprformance 
    #(High increase in casing errors)
    if (speech.shape[0] / rate) < (30 + long_form_add_threshold):
        speech = librosa.util.fix_length(speech, size=(16000 * 30))
        text = model(speech)[0][-2]
    else:
        text = model.decode_long_batched_buffered(
            speech,
            batch_size=batch_size,
            context_len_in_secs=context_len_in_secs,
        )
    #text = model.decode_long_batched_buffered(
    #    speech,
    #    batch_size=batch_size,
    #    context_len_in_secs=context_len_in_secs,
    #)
    return text