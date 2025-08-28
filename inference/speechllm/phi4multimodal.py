from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from utils import read_txt_file


def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
    ).cuda()

    generation_config = GenerationConfig.from_pretrained(model_path)

    return model, processor, generation_config


def generate(model_processor_config, prompt, example_path, modality):
    import soundfile as sf

    if modality != "audio":
        raise NotImplementedError("Phi only supports audio in this implementation!")

    model, processor, generation_config = model_processor_config

    # prompts
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"

    if example_path.endswith(".wav"):
        prompt = f"{user_prompt}<|audio_1|>{prompt}{prompt_suffix}{assistant_prompt}"

        # Open audio file
        audio, samplerate = sf.read(example_path)

        # Process with the model
        inputs = processor(
            text=prompt, audios=[(audio, samplerate)], return_tensors="pt"
        ).to("cuda:0")

    else:
        example = read_txt_file(example_path)
        prompt = f"{user_prompt}{example}\n{prompt}{prompt_suffix}{assistant_prompt}"
        inputs = processor(text=prompt, audios=None, return_tensors="pt").to("cuda:0")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response