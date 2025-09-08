# short-form

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
from datasets import load_dataset, Audio

from tools import build_segtext_to_talkid  # { seg_text: talkid }


# -------------------- Config --------------------
SUB_DIR   = "acl6060-short"
SPLIT     = "eval"
SRC_LANG  = "en"
TGT_LANGS = ["de", "fr", "pt", "zh"]

# -------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# -------------------- Utils --------------------
def get_data_dirs() -> Tuple[Path, Path, Path]:
    root_dir = os.environ.get("H2T_DATADIR")
    if not root_dir:
        raise EnvironmentError("H2T_DATADIR is not set")
    root = Path(root_dir).resolve()

    audio_out_dir = root / SUB_DIR / "audio" / SRC_LANG
    manifest_dir  = Path("manifests") / SUB_DIR
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return root, audio_out_dir, manifest_dir


def norm_seg_key(s: str) -> str:
    """seg/text normalize"""
    return " ".join((s or "").split()).strip()


def build_segtext_index() -> Dict[str, str]:
    """
    tools.build_segtext_to_talkid() 결과(원문 키)를
    정규화 키로 재매핑해서 조회 안정성 확보.
    """
    mapping_raw = build_segtext_to_talkid()  # {seg_text_raw: talkid}
    idx: Dict[str, str] = {}
    for seg_raw, talkid in mapping_raw.items():
        idx[norm_seg_key(seg_raw)] = talkid
    return idx


def write_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------- Main --------------------
def main() -> None:
    # Dirs
    _, audio_out_dir, manifest_dir = get_data_dirs()
    log.info("Audio out:     %s", audio_out_dir)
    log.info("Manifests dir: %s", manifest_dir)

    # HF dataset (decoded audio)
    log.info("Loading dataset ymoslem/acl-6060 split=%s ...", SPLIT)
    ds = load_dataset("ymoslem/acl-6060", split=SPLIT)
    ds = ds.cast_column("audio", Audio(decode=True))
    log.info("Loaded %d samples.", len(ds))

    # seg_text → talkid 
    doc_index = build_segtext_index()

    out_buffers: Dict[str, List[dict]] = {lang: [] for lang in TGT_LANGS}

    # Create records
    miss_doc = 0
    for sample in ds:
        sample_id = sample.get("index")
        src_ref   = sample.get("text_en", "") or ""
        key       = norm_seg_key(src_ref)
        doc_id    = doc_index.get(key, "")

        if not doc_id:
            miss_doc += 1

        # Save audio if missing
        audio_path = audio_out_dir / f"{sample_id}.wav"
        if not audio_path.exists():
            try:
                sf.write(
                    str(audio_path),
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                )
            except Exception as e:
                log.warning("Audio write failed (sample_id=%s): %s", sample_id, e)

        # Base record
        base = {
            "dataset_id": "acl_6060",
            "doc_id": doc_id,
            "sample_id": sample_id,
            "src_audio": f"/{SUB_DIR}/audio/{SRC_LANG}/{sample_id}.wav",
            "src_ref": src_ref,
            "src_lang": SRC_LANG,
            "benchmark_metadata": {
                "context": "short",
                "dataset_type": "longform",
                "subset": SPLIT,
            },
        }

        # Per-target language
        for lang in TGT_LANGS:
            out_buffers[lang].append({
                **base,
                "tgt_ref": sample.get(f"text_{lang}"),
                "tgt_lang": lang,
            })

    #Write JSONL
    for lang, recs in out_buffers.items():
        out_path = manifest_dir / f"en-{lang}.jsonl"
        write_jsonl(recs, out_path)
        log.info("Wrote %d records → %s", len(recs), out_path)

    if miss_doc:
        log.warning("doc_id not found for %d samples (after normalization).", miss_doc)

    log.info("Finished short records.")


if __name__ == "__main__":
    main()