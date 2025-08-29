from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
import torch
import json
import os


def load_model():
    model_path = "CohereForAI/aya-expanse-32b"
    
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


def generate(model_tokenizer_config, prompt, text_path):
    model, tokenizer, generation_config = model_tokenizer_config

    # Read JSONL file
    input_data = []
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            input_data.append(json.loads(line.strip()))
    
    # Process each entry
    output_data = []
    for entry in input_data:
        # Extract text from "output" field
        text_to_translate = entry["output"]
        
        # Create full prompt with the text to translate
        full_prompt = f"{prompt}\n\n{text_to_translate}"
        
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
        
        # Create new entry with all original fields plus translation
        new_entry = entry.copy()
        new_entry["translation"] = translation
        output_data.append(new_entry)
    
    # Create output filename
    base_name = os.path.splitext(text_path)[0]
    output_path = f"{base_name}_llm_translated.jsonl"
    
    # Write output JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    
    return f"Translation completed. Output saved to: {output_path}"
