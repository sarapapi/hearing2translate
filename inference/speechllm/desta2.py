from transformers import AutoModel


def load_model():
    model = AutoModel.from_pretrained(
        "DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True
    ).to("cuda")
    return model


def generate(model, model_input):
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "audio", "content": model_input["sample"]},
        {"role": "user", "content": model_input["prompt"]},
    ]

    generated_ids = model.chat(
        messages, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.9
    )

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response