import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# HARD-CODED CONFIG
DATASET_ID = "uclass2"
SRC_LANG = "en"
TGT_LANGS = ["es", "it", "fr", "de", "nl", "pt"]


CTX_KEYWORDS = {
    "reading": ["reading", "read", "passage"],
    "monolog": ["monolog", "mono"],
    "conversation": ["conversation", "dialog", "dialogue", "conv"],
}

def guess_context_from_path(p):
    """Return context string based on filename/path; default 'unknown'."""
    s = str(p).lower()
    for ctx, kws in CTX_KEYWORDS.items():
        for k in kws:
            if k in s:
                return ctx
    return "unknown"

def guess_participant_id(text):
    m = re.search(r"(?:spk|speaker|child|kid|p|s)\s*[-_]*\s*(\d{1,3})", text, re.I)
    return m.group(1) if m else None

def collect_wavs(src_root):
    return sorted(src_root.rglob("*.wav"))

def build_manifests_by_context(audio_dir, base_dir, overwrite=False):
    src_root = Path(audio_dir).resolve()
    if not src_root.exists() or not src_root.is_dir():
        print(f"ERROR: audio_dir not found or not a directory: {src_root}", file=sys.stderr)
        sys.exit(1)

    # gather all wavs
    wavs = collect_wavs(src_root)
    if not wavs:
        print(f"WARNING: no .wav files found under {src_root}", file=sys.stderr)

    # group by context
    by_ctx = {}
    for wav in wavs:
        ctx = guess_context_from_path(wav)
        by_ctx.setdefault(ctx, []).append(wav)

    total_records = 0

    for ctx, files in sorted(by_ctx.items()):
        # create stage dir for this context
        stage_dir = Path(base_dir) / DATASET_ID / "audio" / ctx
        stage_dir.mkdir(parents=True, exist_ok=True)

        # manifests dir for this context
        manifests_ctx_dir = Path("manifests") / DATASET_ID / ctx
        manifests_ctx_dir.mkdir(parents=True, exist_ok=True)

        # dynamic ID width for this context
        id_width = max(1, len(str(len(files))))

        base_records = []
        for i, src_wav in enumerate(files, start=1):
            sample_id = str(i).zfill(id_width)
            staged_name = src_wav.name  # keep original filename
            dst_wav = stage_dir / staged_name

            # copy file
            try:
                if not dst_wav.exists() or overwrite:
                    dst_wav.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_wav, dst_wav)
            except Exception as e:
                print(f"ERROR copying {src_wav} -> {dst_wav}: {e}", file=sys.stderr)
                continue

            context = ctx
            pid = guess_participant_id(src_wav.name) or guess_participant_id(str(src_wav.parent)) or None

        
            base_record = {
                "dataset_id": DATASET_ID,
                "sample_id": sample_id,
                "src_audio": "/{}/{}/audio/{}/{}".format(DATASET_ID, "", ctx, staged_name).replace("//", "/"),
                "src_ref": None,
                "tgt_ref": None,
                "src_lang": SRC_LANG,
                # tgt_lang inserted later to ensure order
                "benchmark_metadata": {
                    "native_acc": None,
                    "spoken_acc": None,
                    "participant_id": pid,
                    "context": context
                }
            }
            base_records.append(base_record)

        # write per-target manifests for this context
        written_ctx = 0
        for tgt in TGT_LANGS:
            out_path = manifests_ctx_dir / f"{SRC_LANG}-{tgt}.jsonl"
            written = 0
            with out_path.open("w", encoding="utf-8") as outf:
                for rec in base_records:
                    # Rebuild ordered record so tgt_lang is immediately after src_lang
                    ordered = {
                        "dataset_id": rec["dataset_id"],
                        "sample_id": rec["sample_id"],
                        "src_audio": rec["src_audio"],
                        "src_ref": rec["src_ref"],
                        "tgt_ref": rec["tgt_ref"],
                        "src_lang": rec["src_lang"],
                        "tgt_lang": tgt,
                        "benchmark_metadata": rec["benchmark_metadata"],
                    }
                    outf.write(json.dumps(ordered, ensure_ascii=False) + "\n")
                    written += 1
            print(f"Wrote {written} records to {out_path.resolve()}")
            written_ctx += written

        total_records += written_ctx
        print("Staged audio dir for context '{}': {}".format(ctx, stage_dir.resolve()))
        print("Manifest folder for context '{}': {}".format(ctx, manifests_ctx_dir.resolve()))
        print("----")

    print("\nDone.")
    print("Total manifest records written across contexts:", total_records)


def main():
    base_dir = os.environ.get("H2T_DATADIR")
    if not base_dir:
        print("ERROR: environment variable H2T_DATADIR is not set.", file=sys.stderr)
        print("In PowerShell set it like: $env:H2T_DATADIR = 'C:\\path\\to\\folder'", file=sys.stderr)
        sys.exit(1)

    try:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR creating base directory {base_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Stage WAVs by context and build JSONL manifests per-context")
    parser.add_argument("--audio_dir", required=True, help="Directory containing .wav files (recursive search)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite staged audio files if they already exist")
    args = parser.parse_args()

    build_manifests_by_context(args.audio_dir, base_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
