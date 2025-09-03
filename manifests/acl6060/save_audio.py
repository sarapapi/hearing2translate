"""
Save ACL-6060 audio files from a JSONL manifest.

- Audio files are stored under audio/{split}/{sample_id}.wav
- JSONL is used only to determine which sample_ids to export.
"""

import os, json, argparse
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None


def main():
    ap = argparse.ArgumentParser(description="Save ACL-6060 audio to audio/{split}/{sample_id}.wav")
    ap.add_argument("--jsonl", required=True, help="Path to JSONL manifest (with sample_id)")
    ap.add_argument("--split", default="dev", choices=["dev", "eval"], help="Dataset split to use")
    ap.add_argument("--audio_root", default="audio", help="Root directory to save audio")
    args = ap.parse_args()

    # Out directory is audio/{split}
    out_dir = Path(args.audio_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect sample_ids we want to export
    wanted = {}
    for rec in read_jsonl(args.jsonl):
        idx = to_int(rec.get("sample_id"))
        if idx is not None:
            wanted[idx] = True
    if not wanted:
        print("[ERR] No sample_id found in JSONL")
        return

    # 2) Load dataset with decoded audio
    src_dataset = load_dataset("ymoslem/acl-6060", split=args.split)
    src_dataset = src_dataset.cast_column("audio", Audio(decode=True))

    saved, missed = 0, 0
    for sample in src_dataset:
        idx = sample.get("index")
        if idx is None or idx not in wanted:
            continue

        audio_path = out_dir / f"{idx}.wav"
        if audio_path.exists():
            saved += 1
            continue

        try:
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            sf.write(str(audio_path), audio_array, sampling_rate)
            saved += 1
        except Exception as e:
            missed += 1
            if missed <= 5:
                print(f"[MISS_IO] index={idx} error={e}")

    print(f"[DONE] split={args.split} saved={saved} missed={missed} -> {out_dir}")


if __name__ == "__main__":
    main()