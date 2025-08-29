from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch


def load_model():
    processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Small-24B-2507")
    model = VoxtralForConditionalGeneration.from_pretrained(
        "mistralai/Voxtral-Small-24B-2507", torch_dtype=torch.bfloat16, device_map="auto")
    return model, processor


def generate(model_processor, model_input):
    model, processor = model_processor

    conversation = {
        "role": "user",
        "content": [
            {"type": "audio", "path": model_input["sample"]},
            {"type": "text", "text": model_input["prompt"]},
        ],
    }

    inputs = processor.apply_chat_template([conversation])
    inputs = inputs.to(model.device, dtype=torch.bfloat16)

    outputs = model.generate(**inputs, max_new_tokens=4096)
    decoded_outputs = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return decoded_outputs[0]
