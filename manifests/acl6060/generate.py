import os
import json
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio

# --- Configuration ---
LANGUAGE_PAIRS = [
    "en-de",
    "en-fr",
    "en-pt",
    "en-zh",
]

# ACL-6060 provides "dev" and "eval"
SPLITS = ["eval"]


def process_acl6060_dataset():
    """
    Downloads/processes the ACL-6060 dataset (HF: ymoslem/acl-6060) for specified language pairs.
    For each pair and split, it creates a JSONL file with metadata and saves source English audio
    to a local directory.

    Notes:
    - Only English audio is available in this dataset.
    - Target references come from fields like `text_de`, `text_fr`, `text_pt`, `text_zh`.
    """
    print("Starting ACL-6060 dataset processing...")

    # 0) Check required environment variable (audio output root)
    data_root = os.environ.get("H2T_DATADIR")
    if not data_root:
        raise EnvironmentError("H2T_DATADIR is not set")

    # 1) Create output directory for English audio (one shared dir for all pairs)
    audio_output_dir = Path(data_root) / "acl6060" / "audio" / "en"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"English audio will be saved in: '{audio_output_dir}'")
    
    manifest_dir = Path("manifests") / "acl6060"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for pair in LANGUAGE_PAIRS:
        try:
            src_lang, tgt_lang = pair.split("-")
            print(f"\nProcessing language pair: {src_lang} -> {tgt_lang}")

            if src_lang != "en":
                print(f"Warning: Skipping pair '{pair}' (ACL-6060 has English audio only).")
                continue

            for split in SPLITS:
                print(f"--- Processing split: {split} ---")

                # 2) Load dataset once (decoded audio), same for all targets
                print(f"Loading 'ymoslem/acl-6060' split='{split}' with decoded audio...")
                dataset = load_dataset("ymoslem/acl-6060", split=split)
                dataset = dataset.cast_column("audio", Audio(decode=True))
                print(f"Loaded {len(dataset)} samples for split '{split}'.")

                # 3) Prepare JSONL filename per pair & split
                jsonl_filename = Path(manifest_dir) / f"{src_lang}-{tgt_lang}.jsonl"
                records_written = 0

                with open(jsonl_filename, "w", encoding="utf-8") as f:
                    for sample in dataset:
                        sample_id = sample.get("index")

                        # Define audio filename to {index}.wav for consistency
                        fname = f"{sample_id}.wav"

                        audio_filepath = audio_output_dir / fname
                        relative_audio_path = f"/acl6060/audio/en/{fname}"

                        # Save the audio file only if it doesn't already exist
                        if not audio_filepath.exists():
                            sf.write(
                                str(audio_filepath),
                                sample["audio"]["array"],
                                sample["audio"]["sampling_rate"]
                            )

                        # Build target text field name (e.g., 'text_de')
                        tgt_field = f"text_{tgt_lang}"
                        tgt_text = sample.get(tgt_field)
                        if tgt_text is None:
                            if records_written < 5:
                                print(f"Warning: sample_id {sample_id} has no field '{tgt_field}', skipping.")
                            continue

                        # Construct JSON record
                        record = {
                            "dataset_id": "acl_6060",
                            "sample_id": sample_id,
                            "src_audio": relative_audio_path,
                            "src_ref": sample["text_en"],
                            "tgt_ref": tgt_text,
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "benchmark_metadata": {
                                "context": "short",  # segmented utterances
                                "dataset_type": "longform",
                                "doc_id": None, # does not specified in the dataset
                                "subset": split,
                                "orginal_file": sample["audio"]["path"],
                            },
                        }

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records_written += 1

                print(f"Successfully created '{jsonl_filename}' with {records_written} records.")

        except Exception as e:
            print(f"An error occurred while processing pair '{pair}': {e}")

    print("\nDataset processing finished.")


if __name__ == "__main__":
    process_acl6060_dataset()