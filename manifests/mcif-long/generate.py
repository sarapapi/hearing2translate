import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

# -------------------- Config --------------------
ROOT_DIR = os.environ.get("H2T_DATADIR")
if not ROOT_DIR:
    raise EnvironmentError("H2T_DATADIR is not set")

SUB_DIR = "mcif-long"
SRC_LANG = "en"
TGT_LANGS = ["de", "it", "zh"]

MOVE_AUDIO = True

# -------------------- Paths --------------------
ROOT_PATH = Path(ROOT_DIR).resolve()
MANIFEST_DIR = (Path("manifests") / SUB_DIR).resolve()
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_IN_DIR = (ROOT_PATH / SUB_DIR / "audio" / "original").resolve()
AUDIO_OUT_DIR = (ROOT_PATH / SUB_DIR / "audio" / SRC_LANG).resolve()
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_DIR = (ROOT_PATH / SUB_DIR / "long_texts").resolve()
XML_DIR = (ROOT_PATH / SUB_DIR / "xml").resolve()

# -------------------- Writer --------------------
def write_jsonl(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def open_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return " ".join(lines).strip()

# -------------------- Parser --------------------
def parse_xml(xml_path: Path, target_stem: str) -> Optional[str]:
    if not xml_path.exists():
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for sample in root.findall(".//sample[@task='TRANS']"):
        audio = (sample.findtext("audio_path") or "").strip()
        audio_stem = Path(audio).stem
        if audio_stem != target_stem:
            continue

        # metadata/transcript
        meta = sample.find("metadata")
        if meta is not None:
            txt = meta.findtext("reference")
            if txt and txt.strip():
                return txt.strip()

        # transcript
        txt = sample.findtext("reference")
        if txt and txt.strip():
            return txt.strip()

        return None

    return None

# -------------------- Main --------------------
def main() -> None:
    if not TEXT_DIR.exists():
        raise FileNotFoundError(f"TEXT_DIR not found: {TEXT_DIR}")

    buckets: Dict[str, List[dict]] = {lang: [] for lang in TGT_LANGS}

    text_files = sorted([p for p in TEXT_DIR.iterdir() if p.is_file()])
    for idx, file_path in enumerate(text_files):
        print(file_path)
        fname = file_path.stem
        orig_audio_file = (AUDIO_IN_DIR / f"{fname}.wav")

        if not orig_audio_file.exists():
            print(f"[WARN] Original audio not found for '{file_path.name}': {orig_audio_file} (skip audio move)")
            continue 

        new_audio_path = (AUDIO_OUT_DIR / f"{idx}.wav")
        new_audio_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if MOVE_AUDIO:
                if new_audio_path.exists():
                    new_audio_path.unlink()
                shutil.move(str(orig_audio_file), str(new_audio_path))
            else:
                shutil.copy2(orig_audio_file, new_audio_path)
        except Exception as e:
            print(f"[WARN] Failed to write audio: {orig_audio_file} -> {new_audio_path} ({e})")
            continue

        src_ref = open_file(file_path)

        base_record = {
            "dataset_id": "mcif_v1.0",
            "sample_id": idx,
            "src_audio": str(new_audio_path.relative_to(ROOT_PATH)),  # mcif-long/audio/en/0.wav
            "src_ref": src_ref,
            "src_lang": SRC_LANG,
            "benchmark_metadata": {
                "context": "long",
                "dataset_type": "unseen",
                "subset": "test",
                "original_id": fname,
            },
        }

        target_stem = fname
        for tgt_lang in TGT_LANGS:
            tgt_xml = XML_DIR / f"MCIF0.2.IF.long.{tgt_lang}.ref.xml"
            if not tgt_xml.exists():
                print(f"[WARN] Target XML not found for {tgt_lang}: {tgt_xml} (skip)")
                continue

            tgt_ref = parse_xml(tgt_xml, target_stem)
            if not tgt_ref:
                print(f"[INFO] No TRANS transcript for {tgt_lang} / {target_stem} (skip)")
                continue
            tgt_ref = tgt_ref.replace("\n", " ").strip()

            rec = dict(base_record)
            rec["tgt_lang"] = tgt_lang
            rec["tgt_ref"] = tgt_ref
            buckets[tgt_lang].append(rec)

    for tgt_lang, records in buckets.items():
        if not records:
            print(f"[INFO] No records for {tgt_lang}, skip writing.")
            continue
        out_path = MANIFEST_DIR / f"en-{tgt_lang}.jsonl"
        write_jsonl(records, out_path)
        print(f"[OK] Wrote {len(records)} records -> {out_path}")

if __name__ == "__main__":
    main()