import os
import json
from pathlib import Path
from collections import defaultdict
from pydub import AudioSegment

# -------------------- path --------------------
ROOT_DIR = os.environ.get("H2T_DATADIR")
if not ROOT_DIR:
    raise EnvironmentError("H2T_DATADIR is not set")
ROOT_DIR = Path(ROOT_DIR)

ELITR_DIR = ROOT_DIR / "elitr-testset" / "documents" / "iwslt2020-nonnative-slt" / "testset"

SRC_LANG = "en"
TGT_LANG = "de"

# To save files from .mp3/.aac to .wav
WAV_OUT_DIR = ROOT_DIR / "elitr" / "audio" / SRC_LANG
WAV_OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = Path("manifests") / "elitr-long" / f"{SRC_LANG}-{TGT_LANG}.jsonl"
JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------- excluded directory --------------------
# The files of 'khan-academy'and 'khan-academy-new' are similar.
def should_exclude_dir(dname: str) -> bool:
    return dname == "khan-academy"

# -------------------- audio format rules --------------------
# - khan-academy-new: *.en.aac
# - the rest:         *.en.OS.mp3
AUDIO_RULES = {
    "khan-academy-new": {
        "match_func": lambda name: name.endswith(".en.aac"),
        "format": "aac",
    },
}

def audio_match(name: str, subdir_name: str) -> tuple[bool, str]:
    rule = AUDIO_RULES.get(subdir_name)
    if rule:
        return rule["match_func"](name), rule["format"]
    return name.endswith(".en.OS.mp3"), "mp3"

# -------------------- utils --------------------
def stem_key(p: Path) -> str:
    """
    ex) foo.en.OS.mp3 / foo.en.OSt / foo.en.TTde -> foo
    """
    name = p.name
    if ".en." in name:
        return name.split(".en.")[0]
    if name.endswith(".en"):
        return name[:-3]
    return p.stem

def is_src_ref(p: Path) -> bool:
    # source file format: *.en.OSt (not OStt)
    return p.name.endswith(".en.OSt")

def is_tgt_ref(p: Path) -> bool:
    # target file format: *.en.TTde
    return p.name.endswith(f".en.TT{TGT_LANG}")

def build_groups_by_key(subdir_path: Path):
    groups = defaultdict(list)
    for p in subdir_path.iterdir():
        if not p.is_file():
            continue
        # Ignore irrelevant files
        nm = p.name
        if nm.startswith(".") or nm.endswith(".ipycheckpoints") or nm.upper().startswith("README"):
            continue
        groups[stem_key(p)].append(p)
    return groups

def create_empty_tgt_ref(ende_jsonl: Path, tgt_langs: list):
    with ende_jsonl.open("r", encoding="utf-8") as jf:
        records = [json.loads(line) for line in jf if line.strip()]

    all_tgt_langs = ["de", "fr", "pt", "it", "nl", "zh", "es"]

    for lang in all_tgt_langs:
        if lang in tgt_langs:
            continue  

        out_name = ende_jsonl.with_name(ende_jsonl.name.replace("en-de", f"en-{lang}"))
        out_name.parent.mkdir(parents=True, exist_ok=True)

        with out_name.open("w", encoding="utf-8") as f:
            for rec in records:
                rec_out = dict(rec)
                rec_out["tgt_lang"] = lang
                rec_out["tgt_ref"] = "" 
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

        print(f"[INFO] Wrote {len(records)} records → {out_name}")

# -------------------- main process --------------------
def main():
    print(f"Starting ELITR dataset processing for {SRC_LANG} -> {TGT_LANG} ...")
    if not ELITR_DIR.exists():
        raise FileNotFoundError(f"ELITR_DIR not found: {ELITR_DIR}")

    subdirs = sorted(d for d in ELITR_DIR.iterdir() if d.is_dir() and not should_exclude_dir(d.name))

    records_written = 0
    sample_id = 0

    with JSONL_PATH.open("w", encoding="utf-8") as jf:
        for subdir in subdirs:
            file_groups = build_groups_by_key(subdir)

            for key, files in file_groups.items():
                try:
                    audio_file = None
                    audio_format = None
                    src_ref_file = None
                    tgt_ref_file = None

                    has_candidate = False

                    for fp in files:
                        matched, fmt = audio_match(fp.name, subdir.name)
                        if matched:
                            has_candidate = True
                            if audio_file is None:
                                audio_file = fp
                                audio_format = fmt
                            continue

                        if is_src_ref(fp):
                            has_candidate = True
                            if src_ref_file is None:
                                src_ref_file = fp
                            continue

                        if is_tgt_ref(fp):
                            has_candidate = True
                            if tgt_ref_file is None:
                                tgt_ref_file = fp
                            continue

                    if not has_candidate:
                        continue

                    # Create samples when three files all exist
                    if not (audio_file and src_ref_file and tgt_ref_file):
                        missing = []
                        if not audio_file:   missing.append("audio")
                        if not src_ref_file: missing.append("src_ref")
                        if not tgt_ref_file: missing.append("tgt_ref")
                        print(f"[SKIP] '{subdir.name}/{key}' missing: {', '.join(missing)}")
                        continue

                    # ----- convert audio: (mp3 / aac) -> wav -----
                    out_wav_path = WAV_OUT_DIR / f"{sample_id}.wav"
                    audio_seg = AudioSegment.from_file(audio_file, format=audio_format)
                    audio_seg.export(out_wav_path, format="wav")

                    # ----- load text -----
                    with src_ref_file.open(encoding="utf-8") as f:
                        src_ref = " ".join(l.strip() for l in f.readlines())
                    with tgt_ref_file.open(encoding="utf-8") as f:
                        tgt_ref = " ".join(l.strip() for l in f.readlines())

                    # ----- write record -----
                    record = {
                        "dataset_id": "elitr",
                        "sample_id": sample_id,
                        "src_audio": str(out_wav_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                        "src_ref": src_ref,
                        "tgt_ref": tgt_ref,
                        "src_lang": SRC_LANG,
                        "tgt_lang": TGT_LANG,
                        "benchmark_metadata": {
                            "context": "long",
                            "original_file": f"{subdir.name}/{audio_file.name}",
                            "dataset_type": "non_native",
                        },
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                    records_written += 1
                    sample_id += 1

                except Exception as e:
                    print(f"[ERROR] Processing '{subdir.name}/{key}' failed: {e}")

    print(f"\nDataset processing finished. Records written: {records_written}")
    print(f"JSONL: {JSONL_PATH}")
    print(f"WAV out: {WAV_OUT_DIR}")
    
    create_empty_tgt_ref(JSONL_PATH, TGT_LANG)

    
if __name__ == "__main__":
    main()