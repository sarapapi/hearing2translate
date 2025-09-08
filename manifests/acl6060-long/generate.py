# long-form
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List

# -------------------- Import helpers --------------------
CUR_DIR = Path(__file__).resolve().parent
TOOLS_DIR = (CUR_DIR.parent / "acl6060-short").resolve()
sys.path.append(str(TOOLS_DIR))

from tools import build_talkid_to_allsegs  # -> {doc_id: "seg1 seg2 ..."}

# -------------------- Config --------------------
ROOT_DIR = os.environ.get("H2T_DATADIR")
if not ROOT_DIR:
    raise EnvironmentError("H2T_DATADIR is not set")

SUB_DIR = "acl6060-long"
SRC_LANG = "en"
TGT_LANGS = ["de", "fr", "pt", "zh"]

# -------------------- Paths --------------------
ROOT_PATH = Path(ROOT_DIR).resolve()
MANIFEST_DIR = (Path("manifests") / SUB_DIR).resolve()
AUDIO_OUT_DIR = (ROOT_PATH / SUB_DIR / "audio" / SRC_LANG).resolve()
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

XML_EN = MANIFEST_DIR / f"{SRC_LANG}.xml"
MAPPING_FILE = MANIFEST_DIR / "long_audio_mapping.txt"  


# -------------------- Parsers --------------------
def parse_docid_to_sampleid(file_path: Path) -> Dict[str, int]:
    """
    Parse mapping lines of the form:
      2022.acl-long.111\t sample_id=1\t file=2022.acl-long.111.wav

    Returns:
      {doc_id: sample_id(int)}
    """
    mapping: Dict[str, int] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc_id, sample_id_field, *_ = line.strip().split("\t")
            sample_id = int(sample_id_field.split("=", 1)[1])
            mapping[doc_id] = sample_id
    return mapping


def build_lang_doc_map(xml_path: Path) -> Dict[str, str]:
    """
    tools.build_talkid_to_allsegs wrapper: returns {doc_id: "seg1 seg2 ..."}
    xml_path는 manifests/acl6060-long/{lang}.xml
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    return build_talkid_to_allsegs(str(xml_path))


# -------------------- Writer --------------------
def write_jsonl(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------- Main --------------------
def main() -> None:
    # 1) Load mapping: {doc_id: sample_id}
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")
    docid_to_sampleid = parse_docid_to_sampleid(MAPPING_FILE)

    # 2) Load source file: {doc_id: "all segs"}
    docid_to_src = build_lang_doc_map(XML_EN)

    # 3) Load tgt files → create JSONL
    for tgt_lang in TGT_LANGS:
        tgt_xml = MANIFEST_DIR / f"{tgt_lang}.xml"
        if not tgt_xml.exists():
            print(f"[WARN] Target XML not found for {tgt_lang}: {tgt_xml} (skip)")
            continue

        docid_to_tgt = build_lang_doc_map(tgt_xml)

        # Sor by sample_id
        pairs: List[Tuple[str, int]] = sorted(docid_to_sampleid.items(), key=lambda kv: kv[1])

        records: List[dict] = []
        for doc_id, sample_id in pairs:
            src_ref = docid_to_src.get(doc_id)
            tgt_ref = docid_to_tgt.get(doc_id)

            if src_ref is None or tgt_ref is None:
                if src_ref is None:
                    print(f"[WARN] Missing src_ref for doc_id={doc_id} (skip)")
                if tgt_ref is None:
                    print(f"[WARN] Missing tgt_ref for doc_id={doc_id} lang={tgt_lang} (skip)")
                continue

            src_audio_rel = f"/{SUB_DIR}/audio/{SRC_LANG}/{sample_id}.wav"

            rec = {
                "dataset_id": "acl_6060",
                "doc_id": doc_id,
                "sample_id": sample_id,
                "src_audio": src_audio_rel,
                "src_ref": src_ref,
                "tgt_ref": tgt_ref,
                "src_lang": SRC_LANG,
                "tgt_lang": tgt_lang,
                "benchmark_metadata": {
                    "context": "long",
                    "dataset_type": "longform",
                    "subset": "eval",
                },
            }
            records.append(rec)

        out_jsonl = MANIFEST_DIR / f"en-{tgt_lang}.jsonl"
        write_jsonl(records, out_jsonl)
        print(f"[INFO] Wrote {len(records)} records → {out_jsonl}")

    print("[INFO] Finished long-form manifests.")


if __name__ == "__main__":
    main()