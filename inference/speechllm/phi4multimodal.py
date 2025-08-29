from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import soundfile as sf


def load_model():
    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
    ).cuda()

    generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

    return model, processor, generation_config


def generate(model_processor_config, model_input):
    model, processor, generation_config = model_processor_config

    composed_prompt = f"<|user|><|audio_1|>{model_input["prompt"]}<|end|><|assistant|>"

    # Open audio file
    audio, samplerate = sf.read(model_input["sample"])

    # Process with the model
    inputs = processor(
        text=composed_prompt, audios=[(audio, samplerate)], return_tensors="pt"
    ).to("cuda:0")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response
