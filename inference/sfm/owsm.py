from transformers import pipeline
#Requirements are
#librosa
#torch
#espnet
#espnet_model_zoo

import soundfile as sf
import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
from langcodes import Language



def load_model():
    model = Speech2TextGreedySearch.from_pretrained(
        "espnet/owsm_ctc_v4_1B",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_flash_attn=False,
        generate_interctc_outputs=False,
    )
    return model

def generate(model, sample):

    is_long = True # TODO

    context_len_in_secs = 4   # left and right context when doing buffered inference
    batch_size = 1   # Ultra fast, doesnt really matter alot

    src = Language.get(sample["src_lang"]).to_alpha3()
    tgt = Language.get(sample["tgt_lang"]).to_alpha3()
    model.lang_sym=f'<{src}>'
    model.task_sym = "<asr>" if src == tgt else f"<st_{tgt}>"

    speech, rate = sf.read(sample["sample"]
    )

    #TODO Should we distinguish between short-form and long-form for inference?.
    # I.e, is there a case where in short form dataset there is audio >30secs long?
    text = model.decode_long_batched_buffered(
        speech,
        batch_size=batch_size,
        context_len_in_secs=context_len_in_secs,
    )
    print(text)
    return text