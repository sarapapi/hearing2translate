import os
import json
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio

SPLITS = ["test"]
SRC_LANG = "en"

def process_edacc_dataset():
    """
    Process EDACC dataset into dialog-style JSONL:
      - Keep the same doc_id as long as the same two speakers are involved
      - When a new (third) speaker appears, increment doc_id and start a new dialog
    File creation: use 'x' for the first write (prevent overwrite), then 'a' for append
    """
    print("Starting EDACC dataset processing (group by speaker pair)...")

    data_root = os.environ.get("H2T_DATADIR")
    if not data_root:
        raise EnvironmentError("H2T_DATADIR is not set")

    audio_output_dir = Path(data_root) / "edacc" / "audio" / SRC_LANG
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"English audio will be saved in: '{audio_output_dir}'")

    manifest_dir = Path("manifests") / "edacc"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\n--- Processing split: {split} ---")

        ds = load_dataset("edinburghcstr/edacc", split=split)
        ds = ds.cast_column("audio", Audio(decode=True))
        print(f"Loaded {len(ds)} samples for split '{split}'.")

        jsonl_filename = manifest_dir / f"{SRC_LANG}.jsonl"
        mode = "x" if not jsonl_filename.exists() else "a"
        records_written = 0

        # Track speaker pairs
        doc_id = 0
        current_pair = set()   # current dialog speakers (max 2)

        with open(jsonl_filename, mode, encoding="utf-8") as f:
            for i, sample in enumerate(ds):
                # Use provided id if available, else fallback to loop index
                sample_id = sample.get("id", i)

                # Speaker identifier
                speaker = sample.get("speaker")
                if speaker is None:
                    speaker = "UNK"

                # Dialog boundary logic
                if len(current_pair) == 0:
                    # Start a new dialog
                    current_pair = {speaker}
                elif speaker in current_pair:
                    # Continue current dialog
                    pass
                elif len(current_pair) == 1:
                    # Second speaker joins → still same dialog
                    current_pair.add(speaker)
                else:
                    # Third speaker appears → start a new dialog
                    doc_id += 1
                    current_pair = {speaker}

                # Save audio if it does not already exist
                wav_name = f"{sample_id}.wav"
                audio_path = audio_output_dir / wav_name
                rel_audio = f"/edacc/audio/{SRC_LANG}/{wav_name}"

                if not audio_path.exists():
                    sf.write(
                        str(audio_path),
                        sample["audio"]["array"],
                        sample["audio"]["sampling_rate"]
                    )

                # Build JSON record
                record = {
                    "dataset_id": "edacc",
                    "sample_id": sample_id,
                    "src_audio": rel_audio,
                    "src_ref": sample.get("text", ""),
                    "tgt_ref": "",
                    "src_lang": SRC_LANG,
                    "tgt_lang": "",
                    "benchmark_metadata": {
                        "context": "short",          # CHECK!!
                        "dataset_type": "accents",
                        "doc_id": int(doc_id),
                        "speaker": speaker,
                        "accent": sample.get("accent"),
                        "raw_accent": sample.get("raw_accent"),
                        "gender": sample.get("gender"),
                        "l1": sample.get("l1"),
                        "subset": split
                    }
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

        print(f"Wrote {records_written} records to '{jsonl_filename}'.")

    print("\nEDACC processing finished (group by speaker pair).")

if __name__ == "__main__":
    process_edacc_dataset()