import os
import json
import soundfile as sf
from datasets import load_dataset

# --- Configuration ---
# Define the language pairs you want to process.
# The format is 'source_language-target_language'.
# These should be two-letter ISO 639-1 language codes.
LANGUAGE_PAIRS = [
    "en-de",
    "en-fr",
    "en-pt",
    "en-nl",
    "en-it",
    "en-es",
    "en-zh",

    "de-en",
    "fr-en",
    "pt-en",
    "it-en",
    "es-en",
    "zh-en",
]

# This dictionary maps the two-letter language codes to the specific
# configuration names used in the Hugging Face dataset.
LANG_CODE_TO_CONFIG = {
    "de": "de_de",
    "en": "en_us",
    "fr": "fr_fr",
    "es": "es_419",
    "nl": "nl_nl",
    "it": "it_it",
    "zh": "cmn_hans_cn",
    "pt": "pt_br"
}

# Define the splits to be used from the dataset
# The FLEURS dataset on Hugging Face uses 'validation' and 'test'.
SPLITS = ["test"]


def process_fleurs_dataset():
    """
    Downloads and processes the FLEURS dataset for specified language pairs.
    For each pair and each split (dev, test), it creates a .jsonl file
    with metadata and saves the source audio files to a local directory.
    """
    print("Starting FLEURS dataset processing...")

    for pair in LANGUAGE_PAIRS:
        try:
            src_lang, tgt_lang = pair.split('-')
            print(f"\nProcessing language pair: {src_lang} -> {tgt_lang}")

            if src_lang not in LANG_CODE_TO_CONFIG or tgt_lang not in LANG_CODE_TO_CONFIG:
                print(f"Warning: Skipping pair '{pair}'. Language code not found in LANG_CODE_TO_CONFIG.")
                continue

            src_config = LANG_CODE_TO_CONFIG[src_lang]
            tgt_config = LANG_CODE_TO_CONFIG[tgt_lang]

            # Create output directory for audio files (once per source language)
            audio_output_dir = os.path.join(os.environ['H2T_DATADIR'], "fleurs", "audio", src_lang)
            os.makedirs(audio_output_dir, exist_ok=True)
            print(f"Audio files for '{src_lang}' will be saved in: '{audio_output_dir}'")

            # Loop over each split to create separate files
            for split in SPLITS:
                print(f"--- Processing split: {split} ---")

                # 1. Load source and target language datasets for the current split
                print(f"Loading '{src_config}' and '{tgt_config}' datasets for split: {split}")
                src_dataset = load_dataset("google/fleurs", src_config, split=split, trust_remote_code=True)
                tgt_dataset = load_dataset("google/fleurs", tgt_config, split=split, trust_remote_code=True)

                print(f"Loaded {len(src_dataset)} samples for source '{src_lang}' in split '{split}'.")
                print(f"Loaded {len(tgt_dataset)} samples for target '{tgt_lang}' in split '{split}'.")

                # 2. Create a dictionary for the target transcriptions for easy lookup
                tgt_transcriptions = {item['id']: item['raw_transcription'] for item in tgt_dataset}

                # 3. Create and write to the JSONL file
                jsonl_filename = f"{src_lang}-{tgt_lang}.jsonl"
                records_written = 0
                with open(jsonl_filename, 'w', encoding='utf-8') as f:
                    for sample in src_dataset:
                        sample_id = sample['id']

                        if sample_id not in tgt_transcriptions:
                            print(f"Warning: Skipping sample_id {sample_id} - no matching translation found.")
                            continue

                        # Define the audio file path
                        audio_filename = f"{sample_id}.wav"
                        audio_filepath = os.path.join(audio_output_dir, audio_filename)
                        relative_audio_path = os.path.join("fleurs", "audio", src_lang, audio_filename)

                        # Save the audio file only if it doesn't already exist
                        if not os.path.exists(audio_filepath):
                            sf.write(
                                audio_filepath,
                                sample["audio"]["array"],
                                sample["audio"]["sampling_rate"]
                            )

                        # Construct the JSON record
                        record = {
                            "dataset_id": "fleurs",
                            "sample_id": sample_id,
                            "src_audio": f"/{relative_audio_path}",
                            "src_ref": sample["raw_transcription"],
                            "tgt_ref": tgt_transcriptions[sample_id],
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "benchmark_metadata": {
                                "gender": sample["gender"], "context": "short"
                            }
                        }

                        # Write the JSON record to the file
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        records_written += 1

                print(f"Successfully created '{jsonl_filename}' with {records_written} records.")

        except Exception as e:
            print(f"An error occurred while processing pair '{pair}': {e}")

    print("\nDataset processing finished.")

if __name__ == "__main__":
    
    process_fleurs_dataset()