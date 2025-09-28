import json
import librosa
import tempfile
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download
from spire.hubert_labeler import HubertLabeler
from transformers import AutoTokenizer, AutoModelForCausalLM


REQUIRED_SAMPLE_RATE = 16000

# Load the language mapping
mapping_path = Path(__file__).parents[1] / "language_mapping.json"
with open(mapping_path, "r", encoding="utf-8") as f:
    lang_mapper = json.load(f)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "utter-project/SpireFull", padding_side="left", model_max_length=2048
    )
    model = AutoModelForCausalLM.from_pretrained(
        "utter-project/SpireFull", device_map="auto", torch_dtype="bfloat16"
    )
    model.eval()

    kmeans_model_path = hf_hub_download(
        "utter-project/SpireKMeans", filename="kmeans_model"
    )
    labeler = HubertLabeler("facebook/hubert-large-ll60k", kmeans_model_path)
    labeler.to(model.device)
    labeler.eval()

    return model, tokenizer, labeler


def generate(model_tokenizer_labeler, model_input):
    model, tokenizer, labeler = model_tokenizer_labeler

    # Workaround to be sure the loaded audio is 16000 Hz and mono
    audio, _ = librosa.load(model_input["sample"], sr=REQUIRED_SAMPLE_RATE, mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, REQUIRED_SAMPLE_RATE)
        dsu_sequence = labeler.label_wav(tmp.name)[0]

    tgt_lang = lang_mapper[model_input["tgt_lang"]]
    composed_prompt = f"Speech: {dsu_sequence}\n{tgt_lang}:"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": composed_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=5
    )
    outputs = outputs[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return response
