from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
import torch



def load_model(model_name="google/gemma-3-12b-it"):
    """Load the Gemma-3 model.
    
    Args:
        model_name: The Gemma-3 model identifier (default: 'google/gemma-3-12b-it')
    
    Returns:
        Tuple of (model, tokenizer, generation_config)
    
    Raises:
        Exception: If the model is not found on Hugging Face
    """
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, 
        device_map="auto"
    ).eval()
    
    tokenizer = AutoProcessor.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    
    return model, tokenizer, generation_config


def generate(model_tokenizer_config, model_input):
    """Generate text using the Gemma-3 model."""
    model, tokenizer, generation_config = model_tokenizer_config

    # Concatenate prompt and input with newline
    full_prompt = f"{model_input['prompt']}\n{model_input['sample']}"
    
    # For Gemma-3 models using processor with chat template
    messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=4096,
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    translation = tokenizer.decode(generation, skip_special_tokens=True)
    
    return translation.strip()