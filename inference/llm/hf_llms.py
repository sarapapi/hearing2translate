from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_utils import set_seed
import torch

# Set seed for reproducible generation
set_seed(42)


def load_model(model_name):
    """Load the specified Hugging Face text LLM model.
    
    Args:
        model_name: The model identifier (e.g., 'CohereLabs/aya-expanse-32b', 'google/gemma-3-12b-it', 'Unbabel/Tower-Plus-9B', etc.)
    
    Returns:
        Tuple of (model, tokenizer, generation_config)
    
    Raises:
        Exception: If the model is not found on Hugging Face
    """
    # Special handling for Gemma-3 models
    if "gemma-3" in model_name.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, 
            device_map="auto"
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_name)
    else:
        # Try fast tokenizer first, fallback to slow if needed
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        except Exception as e:
            if "ModelWrapper" in str(e):
                print(f"Fast tokenizer failed for {model_name}, using slow tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            else:
                raise
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    
    # Set pad token if not present (skip for processors)
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # GenerationConfig is optional
    generation_config = None
    try:
        generation_config = GenerationConfig.from_pretrained(model_name)
    except Exception:
        pass  # Not all models have generation configs
    
    return model, tokenizer, generation_config


def generate(model_tokenizer_config, model_input):
    """Generate text using the loaded model."""
    model, tokenizer, generation_config = model_tokenizer_config

    # Concatenate prompt and input with newline
    full_prompt = f"{model_input['prompt']}\n{model_input['sample']}"
    
    # Check if this is a Gemma-3 model (using processor)
    if hasattr(tokenizer, 'apply_chat_template'):
        # For Gemma-3 models using processor
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
                max_new_tokens=512,
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        translation = tokenizer.decode(generation, skip_special_tokens=True)
    else:
        # For other models using regular tokenizer
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda:0")
        
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Extract only the generated part (remove input prompt)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        translation = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    return translation.strip()

