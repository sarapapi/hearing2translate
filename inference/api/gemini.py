from google import genai
import os

MODEL_NAME = "gemini-2.5-flash"

def load_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    return MODEL_NAME, client

def generate(model_processor, model_input):
    model, client = model_processor
    audio_file = client.files.upload(file=model_input["sample"])

    contents = [
        audio_file,
        model_input["prompt"],
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    return response.text.strip()
