"""Download EmotionTalk audio and build manifest files."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_ID = "emotiontalk"
SRC_LANG = "zh"
TGT_LANG = "en"


def process_emotiontalk_dataset() -> None:
    """Download EmotionTalk audio, stage WAVs, and create JSONL manifest."""

    base_dir_env = os.environ.get("H2T_DATADIR")
    if not base_dir_env:
        raise EnvironmentError("H2T_DATADIR environment variable is not set")

    dataset_dir = Path(base_dir_env) / DATASET_ID
    raw_dir = dataset_dir / "raw"
    audio_stage_dir = dataset_dir / "audio" / SRC_LANG
    manifest_path = dataset_dir / f"{SRC_LANG}-{TGT_LANG}.jsonl"
    script_manifest_path = Path(__file__).resolve().parent / f"{SRC_LANG}-{TGT_LANG}.jsonl"

    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_stage_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading EmotionTalk snapshot (audio tar)...")
    snapshot_download(
        repo_id="BAAI/Emotiontalk",
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=["Audio.tar"],
    )

    audio_tar_path = raw_dir / "Audio.tar"
    if not audio_tar_path.is_file():
        raise FileNotFoundError(f"Expected audio tar not found at {audio_tar_path}")

    print("Processing archive and staging audio files...")

    records_written = 0
    with tarfile.open(audio_tar_path, mode="r:*") as tar, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_file:
        members = {member.name: member for member in tar.getmembers()}
        json_members = sorted(
            (m for m in members.values() if m.name.endswith(".json")),
            key=lambda m: m.name,
        )

        for json_member in json_members:
            with tar.extractfile(json_member) as json_fp:  # type: ignore[arg-type]
                if json_fp is None:
                    continue
                metadata = json.load(json_fp)

            file_path = metadata.get("file_path")
            if not file_path:
                continue

            top_level = file_path.split("/", 1)[0]
            if top_level not in {"G00003", "G00015"}:
                continue

            audio_member_name = f"Audio/wav/{file_path}"
            audio_member = members.get(audio_member_name)
            if audio_member is None:
                print(f"Warning: missing audio for {audio_member_name}")
                continue

            sample_id = Path(file_path).stem
            staged_audio_path = audio_stage_dir / f"{sample_id}.wav"

            if not staged_audio_path.exists():
                staged_audio_path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(audio_member) as audio_fp:
                    if audio_fp is None:
                        print(f"Warning: failed to extract audio for {audio_member_name}")
                        continue
                    with staged_audio_path.open("wb") as dst_fp:
                        shutil.copyfileobj(audio_fp, dst_fp)

            record = {
                "dataset_id": DATASET_ID,
                "sample_id": sample_id,
                "src_audio": f"{DATASET_ID}/audio/{SRC_LANG}/{staged_audio_path.name}",
                "src_ref": metadata.get("content"),
                "tgt_ref": None,
                "src_lang": SRC_LANG,
                "tgt_lang": TGT_LANG,
                "benchmark_metadata": {
                    "context": "short",
                    "emotion": metadata.get("emotion_result"),
                    "dataset_type": "emotion",
                },
            }

            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    #shutil.copy2(manifest_path, script_manifest_path)
    print(
        f"Successfully created '{manifest_path}' with {records_written} records.\n"
        "Dataset processing finished."
    )


if __name__ == "__main__":
    process_emotiontalk_dataset()

