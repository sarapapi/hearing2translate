import requests
import base64
import os
import sys
import json

class OpenRouterClient:

    url = "https://openrouter.ai/api/v1/chat/completions"

    KEY_VARIABLE = "OPENROUTER_API_KEY"

    def __init__(self, model_name):
        self.api_key = os.environ.get(self.KEY_VARIABLE)
        if not self.api_key:
            raise RuntimeError(f"{self.KEY_VARIABLE} environment variable not set.")
        self.model_name = model_name

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def encode_audio_to_base64(audio_path):
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')

    def process(self, audio_path, prompt):
        print(f"INFO Processing audio: {audio_path} with prompt: {prompt}", file=sys.stderr)
        base64_audio = self.encode_audio_to_base64(audio_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
        payload = {
            "model": self.model_name,
            "messages": messages
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        return response

def load_model():
    client = OpenRouterClient(model_name="openai/gpt-audio")
    return client

def generate(client, model_input):
    r = client.process(audio_path=model_input["sample"], prompt=model_input["prompt"])
    r = r.json()
    print("INFO RESPONSE:",json.dumps(r), file=sys.stderr)
    return r["choices"][0]["message"]["content"]
