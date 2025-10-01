import os
import json
import shutil
import tarfile
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download


def process_cs_dialogue_dataset():
    """
    Downloads and processes the CS-Dialogue dataset.
    For each pair, it creates a .jsonl file with metadata and saves the source audio files to a local directory.
    """
    print("Starting CS-Dialogue dataset processing...")

    assert os.environ["H2T_DATADIR"] is not None, "H2T_DATADIR is not set"

    dataset_path = Path(os.environ["H2T_DATADIR"]) / "cs-dialogue"
    raw_dataset_path = dataset_path / "raw"
    raw_dataset_path.mkdir(parents=True, exist_ok=True)

    # 1. Download dataset and extract tar files
    snapshot_download(
        repo_id="BAAI/CS-Dialogue",
        repo_type="dataset",
        local_dir=raw_dataset_path,
        allow_patterns=["data/index/short_wav/*", "data/short_wav/*"],
    )
    data_dir = raw_dataset_path / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            target = raw_dataset_path / item.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(item), str(target))
        data_dir.rmdir()

    short_wav_path = raw_dataset_path / "short_wav"
    merged_tar_file = raw_dataset_path / "short_wav.tar.gz"
    if not (short_wav_path / "WAVE").is_dir():
        print("Extracting tar files...")
        if merged_tar_file.is_file():
            os.unlink(merged_tar_file)
        part_files = sorted(short_wav_path.glob("*.tar.gz*"), key=lambda p: p.name)
        with open(merged_tar_file, "wb") as out:
            for fname in part_files:
                with open(fname, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        out.write(chunk)
        shutil.rmtree(short_wav_path)

        with tarfile.open(merged_tar_file, "r:gz") as tar:
            tar.extractall(raw_dataset_path)
        os.unlink(merged_tar_file)

    # 2. Load CS-Dialogue test data into a pandas DataFrame
    text_file = raw_dataset_path / "index" / "short_wav" / "test" / "text"
    wav_scp_file = raw_dataset_path / "index" / "short_wav" / "test" / "wav.scp"
    script_files_dir = raw_dataset_path / "short_wav" / "SCRIPT"

    text_data = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)  # Split on first space only
            if len(parts) == 2:
                sample_id, transcription = parts
                text_data.append(
                    {"sample_id": sample_id, "transcription": transcription}
                )

    audio_data = []
    with open(wav_scp_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)  # Split on first space only
            if len(parts) == 2:
                sample_id, audio_path = parts
                audio_data.append({"sample_id": sample_id, "audio_path": audio_path})

    script_data = []
    for script_file in script_files_dir.iterdir():
        with open(script_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    sample_id, sample = parts
                    script = sample.split(" ")[0]
                    script_data.append({"sample_id": sample_id, "script": script})

    text_df = pd.DataFrame(text_data)
    audio_df = pd.DataFrame(audio_data)
    script_df = pd.DataFrame(script_data)
    df = pd.merge(text_df, audio_df, on="sample_id", how="inner")
    df = pd.merge(df, script_df, on="sample_id", how="inner")

    # 3. Filter out English-only and Chinese-only samples
    df = df[df["script"] == "<MIX>"].drop(columns="script")

    # 4. Add more information to the DataFrame
    df = df.assign(
        dataset_id="cs-dialogue",
        tgt_ref=None,
        src_lang="zh",
        tgt_lang="en",
    )
    df["benchmark_metadata"] = df.apply(
        lambda row: {
            "cs_lang": ["en", "zh"],
            "context": "short",
            "dataset_type": "code_switch",
        },
        axis=1,
    )
    df = df.rename(columns={"audio_path": "src_audio", "transcription": "src_ref"})
    df = df[
        [
            "dataset_id",
            "sample_id",
            "src_audio",
            "src_ref",
            "tgt_ref",
            "src_lang",
            "tgt_lang",
            "benchmark_metadata",
        ]
    ]

    # 5. Copy audio files to the new location
    for i, row in df.iterrows():
        old_src_audio_path = raw_dataset_path / row["src_audio"]
        new_src_audio_path = dataset_path / "audio" / row["src_lang"] / old_src_audio_path.name
        new_src_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(old_src_audio_path, new_src_audio_path)
        df.at[i, "src_audio"] = f"cs-dialogue/audio/{row['src_lang']}/{old_src_audio_path.name}"

    # 6. Write to JSONL file
    jsonl_filename = dataset_path / "zh-en.jsonl"

    records_written = 0
    with open(jsonl_filename, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            records_written += 1

    shutil.copy(jsonl_filename, Path(__file__).parent / "zh-en.jsonl")
    print(
        f"Successfully created '{jsonl_filename}' with {records_written} records."
    )

    print("\nDataset processing finished.")


if __name__ == "__main__":
    process_cs_dialogue_dataset()
