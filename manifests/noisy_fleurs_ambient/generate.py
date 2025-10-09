import os
import json
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset

def process_noisy_fleurs_dataset():
    """
    Downloads and processes the Noisy-FLEURS dataset.
    It creates a .json file for each language pair and stores the .wav files in a local directory.
    """
    print("Starting noisy FLEURS export...")
    print("Dataset: maikezu/noisy-fleurs, split: test, noise type: ambient")
    # Load Hugging Face dataset
    dataset = load_dataset("maikezu/noisy-fleurs", "ambient", split="test", trust_remote_code=True)

    # Base output directories
    h2t_datadir = os.environ['H2T_DATADIR']
    base_audio_dir = os.path.join(h2t_datadir, "noisy_fleurs_ambient", "audio")
    jsonl_output_dir = os.getcwd()
    os.makedirs(jsonl_output_dir, exist_ok=True)

    # Keep track of file handles per language pair
    jsonl_files = {}

    for sample in tqdm(dataset):
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]
        sample_id = sample["sample_id"]

        # --- Audio directory per source language ---
        audio_output_dir = os.path.join(base_audio_dir, src_lang)
        os.makedirs(audio_output_dir, exist_ok=True)

        # --- Save audio ---
        audio_array = sample["src_audio"]["array"]
        sr = sample["src_audio"]["sampling_rate"]
        audio_filename = sample["src_audio"]["path"]
        audio_filepath = os.path.join(audio_output_dir, audio_filename)
        if not os.path.exists(audio_filepath):
            sf.write(audio_filepath, audio_array, sr)

        # --- JSONL file per language pair ---
        langpair_key = f"{src_lang}-{tgt_lang}"
        if langpair_key not in jsonl_files:
            jsonl_path = os.path.join(jsonl_output_dir, f"{langpair_key}.jsonl")
            jsonl_files[langpair_key] = open(jsonl_path, "w", encoding="utf-8")

        # --- Write JSON record ---
        relative_audio_path = os.path.relpath(audio_filepath, start=h2t_datadir)
        record = {
            "dataset_id": sample["dataset_id"],
            "sample_id": sample_id,
            "src_audio": f"/{relative_audio_path}",
            "src_ref": sample["src_ref"],
            "tgt_ref": sample["tgt_ref"],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "benchmark_metadata": sample.get("benchmark_metadata", {})
        }
        jsonl_files[langpair_key].write(json.dumps(record, ensure_ascii=False) + "\n")

    # Close all JSONL files
    for f in jsonl_files.values():
        f.close()

    print("Done! Audio and JSONL files written for all language pairs.")
if __name__ == "__main__":
    process_noisy_fleurs_dataset()