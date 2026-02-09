from google import genai
from google.genai import types  # Added this import
import os

MODEL_NAME = "gemini-2.5-flash"

def load_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        seed=42,           
    )

    return config, client

def generate(model_processor, model_input):
    config, client = model_processor
    audio_file = client.files.upload(file=model_input["sample"])

    contents = [
        audio_file,
        model_input["prompt"],
    ]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )
    if response.text is None:
        return ""
    return response.text.strip()
