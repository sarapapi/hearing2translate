from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa


def load_model():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    return model, processor


def generate(model_processor, model_input):
    model, processor = model_processor

    prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{model_input["prompt"]}"
    audio, sr = librosa.load(model_input["sample"], sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)

    generate_ids = model.generate(**inputs, max_length=4096)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response
