import argparse
import logging
import importlib
import json
import os
from tqdm import tqdm
from transformers import set_seed

set_seed(42)

MODELS = [
    "aya-expanse-32b",
    "canary-v2",
    "gemma3-",
    "phi4multimodal",
    "towerplus"
]

MODEL_MODULES = {
    # llms
    "aya-expanse-32b": "inference.llm.aya",

    # speech foundation models
    "canary-v2": "inference.sfm.canaryv2",

    # speechllms
    "phi4multimodal": "inference.speechllm.phi4multimodal",
}

TEMPLATED_TEXT_PROMPT = \
    ("You are a professional {src_lang}-to-{tgt_lang} translator. Your goal is to accurately convey "
     "the meaning and nuances of the original {src_lang} text while adhering to {tgt_lang} grammar, "
     "vocabulary, and cultural sensitivities. Preserve the line breaks. Use precise terminology "
     "and a tone appropriate for academic or instructional materials. Produce only the {tgt_lang} "
     "translation, without any additional explanations or commentary. Please translate the "
     "provided {src_lang} text into {tgt_lang}:")

TEMPLATED_SPEECH_PROMPT = \
    ("You are a professional {src_lang}-to-{tgt_lang} translator. Your goal is to accurately convey "
     "the meaning and nuances of the original {src_lang} speech while adhering to {tgt_lang} "
     "grammar, vocabulary, and cultural sensitivities. Use precise terminology and a tone "
     "appropriate for academic or instructional materials. Produce only the {tgt_lang} "
     "translation, without any additional explanations or commentary. Please translate the "
     "provided {src_lang} speech into {tgt_lang}:")


def setup_model(model_name):
    if model_name not in MODEL_MODULES:
        raise NotImplementedError(f"Model {model_name} currently not supported!")

    module_name = MODEL_MODULES[model_name]
    module = importlib.import_module(module_name)

    load_func = getattr(module, "load_model", None)
    if not load_func:
        raise ImportError(f"Module {module_name} does not define `load_model`")

    generate_func = getattr(module, "generate", None)
    if not generate_func:
        raise ImportError(f"Module {module_name} does not define `generate`")

    model = load_func()
    return model, generate_func


def load_prompt(modality: str, src_lang: str, tgt_lang: str) -> str:
    """
    Load and fill the prompt template based on modality and language mapping.

    Args:
        modality: either "speech" or "text"
        src_lang: source language code (e.g., 'en')
        ref_lang: target language code (e.g., 'es')

    Returns:
        str: prompt with placeholders replaced
    """
    # Select the prompt based on modality
    if modality == "speech":
        prompt = TEMPLATED_SPEECH_PROMPT
    elif modality == "text":
        prompt = TEMPLATED_TEXT_PROMPT

    # Load the language mapping
    mapping_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference", "language_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        lang_mapper = json.load(f)

    # Replace placeholders
    try:
        filled_prompt = (
            prompt
            .replace("{src_lang}", lang_mapper[src_lang])
            .replace("{tgt_lang}", lang_mapper[tgt_lang])
        )
    except KeyError as e:
        raise ValueError(f"Language not found in mapping: {e}") from e

    return filled_prompt


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_model_input(modality, example, transcripts):
    if modality == "text":
        try:
            transcript = transcripts.get((example["dataset_id"], example["sample_id"]))
        except:
            raise ValueError(
                f'No transcript found for {example["dataset_id"]}/{example["sample_id"]}, but '
                f'modality is {modality}.')
        return transcript
    else:
        return example.get("src_audio")


def infer(args):
    logging.info(f"Loading model {args.model}")
    model, generate = setup_model(args.model)
    modality = args.in_modality

    transcripts = None
    if args.in_modality == "text":
        logging.info("Loading transcripts")
        transcripts = {}
        with open(args.transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Use a tuple of (dataset_id, sample_id) as the key
                key = (entry["dataset_id"], entry["sample_id"])
                transcripts[key] = entry["output"]  # Store the 'output' field as the value

    results = []
    for sample in tqdm(read_jsonl(args.in_file), desc="Generating Outputs"):
        src_lang = sample.get("src_lang")
        tgt_lang = sample.get("tgt_lang")
        context = sample.get("benchmark_metadata")["context"]
        prompt = load_prompt(modality, src_lang, tgt_lang)

        sample_in = get_model_input(modality, sample, transcripts)
        model_input = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "prompt": prompt,   # prompt to be used by SpeechLLMs and LLMs
            "sample": sample_in,    # either the audio path or the transcript to be translated
            "context": context,  # "short" or "long" for short- or long-form
        }

        output = generate(model, model_input).strip()

        results.append({
            "dataset_id": sample["dataset_id"],
            "sample_id": sample["sample_id"],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "output": output
        })

    logging.info(f"Writing results")
    write_jsonl(args.out_file, results)
    logging.info("Output written to %s", args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hearing to Translate output generation.")
    parser.add_argument("--model", choices=MODELS, required=True,
                        help="Model to be used for inference")
    parser.add_argument("--in-modality", choices=["speech", "text"], required=True,
                        help="Input modality used for inference")
    parser.add_argument("--in-file", required=True, help="Input JSONL file path")
    parser.add_argument("--out-file", required=True, help="Output JSONL file path")
    parser.add_argument("--transcript-file",
                        help="Optional JSONL with transcripts for text modality")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    infer(args)
