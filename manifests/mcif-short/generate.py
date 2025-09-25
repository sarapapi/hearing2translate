import os
import json
from pathlib import Path
from typing import List

# -------------------- Config --------------------
ROOT_DIR = os.environ.get("H2T_DATADIR")
if not ROOT_DIR:
    raise EnvironmentError("H2T_DATADIR is not set")

SUB_DIR = "mcif-short"
SRC_LANG = "en"
TGT_LANGS = ["de", "it", "zh"]

# -------------------- Paths --------------------
ROOT_PATH = Path(ROOT_DIR).resolve()
MANIFEST_DIR = (Path("manifests") / SUB_DIR).resolve()
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_OUT_DIR = (ROOT_PATH / SUB_DIR / "audio" / SRC_LANG).resolve()
AUDIO_PATH = (Path("manifests") / "mcif" / "audio_path").resolve()

# -------------------- Writer --------------------
def write_jsonl(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------------------- Main --------------------
def main() -> None:
    for lang in TGT_LANGS:
        audio_path_file = AUDIO_PATH / f"{SRC_LANG}-{lang}.json"
        with open(audio_path_file, "r", encoding="utf-8") as f:
            audio_path_dic = json.load(f)

        sample_id = 0
        records = []
        for old_path, new_dic in audio_path_dic.items():
            new_path = AUDIO_OUT_DIR / new_dic["audio_path"]  # 0.wav

            record = {
                "dataset_id": "mcif_v1.0",
                "sample_id": sample_id,
                "doc_id": new_dic["doc_id"],
                "src_audio": str(new_path.relative_to(ROOT_PATH)),
                "src_lang": SRC_LANG,
                "tgt_lang": lang,
                "benchmark_metadata": {
                    "context": "short",
                    "dataset_type": "unseen",
                    "subset": "test",
                    "original_id": old_path,
                },
            }
            records.append(record)
            sample_id += 1

        out_path = MANIFEST_DIR / f"en-{lang}.jsonl"
        write_jsonl(records, out_path)
        print(f"[OK] Wrote {len(records)} records -> {out_path}")

if __name__ == "__main__":
    main()