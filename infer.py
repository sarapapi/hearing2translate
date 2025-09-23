import argparse
import logging
import importlib
import json
import os
from tqdm import tqdm
from transformers import set_seed
import sys

set_seed(42)

MODEL_MODULES = {
    # speech foundation models
    "canary-v2": "inference.sfm.canaryv2",
    "whisper": "inference.sfm.whisper",
    "seamlessm4t": "inference.sfm.seamlessm4t",

    # speechllms
    "desta2-8b": "inference.speechllm.desta2",
    "phi4multimodal": "inference.speechllm.phi4multimodal",
    "qwen2audio-7b": "inference.speechllm.qwen2audio",
    "voxtral-small-24b": "inference.speechllm.voxtral"
}

MODELS = sorted(list(MODEL_MODULES.keys()))

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


def setup_model(model_name, modality):
    # For text modality, use the unified HuggingFace LLMs module
    if modality == "text":
        module = importlib.import_module("inference.llm.hf_llms")
        
        load_func = getattr(module, "load_model", None)
        if not load_func:
            raise ImportError("HF LLMs module does not define `load_model`")

        generate_func = getattr(module, "generate", None)
        if not generate_func:
            raise ImportError("HF LLMs module does not define `generate`")

        # Pass the model name to the hf_llms module - let it validate and handle
        model = load_func(model_name)
        return model, generate_func
    
    # For speech modality, validate model is in supported list
    if model_name not in MODEL_MODULES:
        raise NotImplementedError(f"Model {model_name} currently not supported for {modality} modality! Supported models: {', '.join(MODELS)}")

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
        return os.path.join(os.environ['H2T_DATADIR'], example.get("src_audio").lstrip(os.sep))


def infer(args):
    logging.info(f"Loading model {args.model}")
    modality = args.in_modality
    model, generate = setup_model(args.model, modality)

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


    inlines = list(read_jsonl(args.in_file))
    if vars(args)["continue"]:
        if not args.out_file:
            raise ValueError("To continue, --out-file must be set.")
        if not os.path.exists(args.out_file):
            outlines = []
        else:
            with open(args.out_file, "r") as f:
                outlines = f.readlines()
        logging.info(f"Continuing from {args.out_file}, skipping {len(outlines)} already processed inputs.")
        inlines = inlines[len(outlines):]
        outfile = open(args.out_file, "a", encoding="utf-8")
    elif args.out_file:
        outfile = open(args.out_file, "w", encoding="utf-8")
    else:
        outfile = sys.stdout
    for sample in tqdm(inlines, desc="Generating Outputs"):
        src_lang = sample.get("src_lang")
        if args.asr:
            tgt_lang = src_lang
        else:
            tgt_lang = sample.get("tgt_lang")
        context = sample.get("benchmark_metadata")["context"]
        prompt = load_prompt(modality, src_lang, tgt_lang)

        sample_in = get_model_input(modality, sample, transcripts)
        model_input = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "prompt": prompt,  # prompt to be used by SpeechLLMs and LLMs
            "sample": sample_in,  # either the audio path or the transcript to be translated
            "context": context,  # "short" or "long" for short- or long-form
        }

        output = generate(model, model_input).strip()

        result = {
            "dataset_id": sample["dataset_id"],
            "sample_id": sample["sample_id"],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "output": output
        }
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        outfile.flush()
    if args.out_file:
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hearing to Translate output generation.")
    parser.add_argument("--model", required=True,
                        help="Model to be used for inference. For text modality: any HuggingFace model name. For speech modality: " + ", ".join(MODELS))
    parser.add_argument("--in-modality", choices=["speech", "text"], required=True,
                        help="Input modality used for inference")
    parser.add_argument("--in-file", required=True, help="Input JSONL file path")
    parser.add_argument("--out-file", required=False, help="Output JSONL file path. If not set: stdout.", default=None)
    parser.add_argument("--transcript-file",
                        help="Optional JSONL with transcripts for text modality")
    parser.add_argument("--asr", default=False, action="store_true",
                        help="If set, the speech model is used as ASR for the src lang. Tgt language is ignored.")
    parser.add_argument("--continue", default=False, action="store_true",
                        help="If set, append new outputs to existing out-file, skipping already processed inputs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    infer(args)
