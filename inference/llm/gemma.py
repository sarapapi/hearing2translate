from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
import torch


def load_model():
    model_path = "google/gemma-2-27b-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    ).cuda()
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generation_config = GenerationConfig.from_pretrained(model_path)

    return model, tokenizer, generation_config


def generate(model_tokenizer_config, prompt, model_input):
    model, tokenizer, generation_config = model_tokenizer_config

    # Concatenate prompt and input with newline
    full_prompt = f"{prompt}\n{model_input}"
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda:0")
    
    # Generate translation
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        generation_config=generation_config,
    )
    
    # Extract only the generated part (remove input prompt)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    translation = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return translation
