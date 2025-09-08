"""
Generate ACL-6060 manifests from HF (short-form) and manual download (long-form).
Uses an existing long_audio_mapping.txt (if present) to map doc_id -> actual long wav filename (e.g. 416.wav).
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import soundfile as sf
from datasets import load_dataset, Audio

ROOT_DIR = os.environ.get("H2T_DATADIR")
if not ROOT_DIR:
    raise EnvironmentError("H2T_DATADIR is not set")
        
# -------------------- Config --------------------
SPLIT = "eval"
SRC_LANG = "en"
TGT_LANGS = ["de", "fr", "pt", "zh"]

# -------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------- XML loader ----------------
_BAD_CTRL = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
_BARE_AMP = re.compile(r'&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]+;)')

def load_xml_root(xml_path: Path) -> ET.Element:
    """
    Load XML with mild sanitization:
    - Remove invalid control chars
    - Escape bare '&'
    """
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    text = _BAD_CTRL.sub("", text)
    text = _BARE_AMP.sub("&amp;", text)
    return ET.fromstring(text)

def doc_to_seg_pairs(xml_path: Path) -> Dict[str, List[Tuple[int, str]]]:
    """
    Convert XML into: { docid: [(seg_id, seg_text), ...] } sorted by seg_id.
    Expected structure: <doc docid="..."><seg id="...">TEXT</seg>...</doc>
    """
    root = load_xml_root(xml_path)
    out: Dict[str, List[Tuple[int, str]]] = {}
    for doc in root.findall(".//doc"):
        docid = doc.attrib.get("docid")
        if not docid:
            continue
        bucket = out.setdefault(docid, [])
        for seg in doc.findall("seg"):
            sid = seg.attrib.get("id")
            if sid is None:
                continue
            try:
                seg_id = int(sid)
            except ValueError:
                continue
            seg_text = (seg.text or "").strip()
            bucket.append((seg_id, seg_text))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out

def build_seg_to_doc_from_en(en_docs: Dict[str, List[Tuple[int, str]]]) -> Dict[int, str]:
    """
    From English doc->[(seg_id, text)] build a global seg_id -> doc_id map.
    """
    seg_to_doc: Dict[int, str] = {}
    for docid, pairs in en_docs.items():
        for seg_id, _ in pairs:
            seg_to_doc[seg_id] = docid
    return seg_to_doc

def lookup_doc_id(seg_to_doc: Dict[int, str], sample_id: int) -> Optional[str]:
    """
    HF sample_id corresponds to XML seg_id + 1.
    """
    return seg_to_doc.get(sample_id - 1)

def get_last_sample_id(jsonl_path: Path) -> int:
    """
    Return the last sample_id in the JSONL, or -1 if missing/empty/unreadable.
    Robust against trailing newlines and malformed lines.
    """
    if not jsonl_path.exists():
        return -1
    last_id = -1
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sid = rec.get("sample_id")
                    if isinstance(sid, int):
                        last_id = sid
                    elif isinstance(sid, str) and sid.isdigit():
                        last_id = int(sid)
                except Exception:
                    continue
    except Exception:
        return -1
    return last_id

# -------------------- Long audio mapping --------------------
def load_long_audio_mapping(path: Path) -> Dict[str, str]:
    """Load mapping: {doc_id: sample_id.wav}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            doc_id, sample_id_part, file_part = parts
            try:
                sample_id = sample_id_part.split("=")[1]
                mapping[doc_id] = f"{sample_id}.wav"
            except IndexError:
                continue
    return mapping

def mapping_doc_id(en_docs, manifest_dir):
    user_mapping_path = Path(manifest_dir) / "long_audio_mapping.txt"
    existing_map = load_long_audio_mapping(user_mapping_path)

    # docid_to_sampleid and docid_to_filename (actual file used, eg "416.wav")
    docid_to_sampleid: Dict[str, int] = {}
    docid_to_filename: Dict[str, str] = {}

    # If existing_map provided, fill those in first and compute next cur_id appropriately
    used_ids = set()
    for docid, fname in existing_map.items():
        stem = Path(fname).stem
        try:
            sid = int(stem)
            docid_to_sampleid[docid] = sid
            docid_to_filename[docid] = fname  # use filename as-is (e.g., "416.wav")
            used_ids.add(sid)
        except Exception:
            # if filename doesn't parse to int, we still set filename but sample id will be assigned later
            docid_to_filename[docid] = fname

    # Ensure docid_to_sampleid matches docid_to_filename where possible (safety)
    for docid, fname in docid_to_filename.items():
        stem = Path(fname).stem
        if stem.isdigit():
            sid = int(stem)
            # if sample id differs from our assigned one, overwrite assigned to follow filename
            if docid_to_sampleid.get(docid) != sid:
                logger.info("Adjusting sample_id for %s to match mapping file filename %s", docid, fname)
                docid_to_sampleid[docid] = sid
    return docid_to_sampleid, docid_to_filename

# -------------------- Main ----------------------
def main():
    # Directories
    audio_out_dir = Path(ROOT_DIR) / "acl6060" / "audio" / SRC_LANG
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path("manifests") / "acl6060"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    xml_dir = manifest_dir / "long"  # contains en.xml and <lang>.xml

    logger.info("Audio out:     %s", audio_out_dir)
    logger.info("Manifests dir: %s", manifest_dir)
    logger.info("XML dir:       %s", xml_dir)

    # Load HF dataset (decoded audio)
    logger.info("Loading dataset ymoslem/acl-6060 split=%s ...", SPLIT)
    ds = load_dataset("ymoslem/acl-6060", split=SPLIT)
    ds = ds.cast_column("audio", Audio(decode=True))
    logger.info("Loaded %d samples.", len(ds))

    # Build doc/seg maps from XML
    en_xml = xml_dir / "en.xml"
    if not en_xml.exists():
        raise FileNotFoundError(f"Missing English XML: {en_xml}")
    en_docs = doc_to_seg_pairs(en_xml)                  # {docid: [(seg_id, en_text), ...]}
    seg_to_doc = build_seg_to_doc_from_en(en_docs)      # seg_id -> docid

    # Build text maps by sample_id (dataset index)
    text_map_en: Dict[int, str] = {}
    text_map_tgt: Dict[str, Dict[int, str]] = {lang: {} for lang in TGT_LANGS}
    max_sample_id = -1

    # -------- write SHORT records (and save short audio as {sample_id}.wav) --------
    for sample in ds:
        sample_id = sample.get("index")
        if isinstance(sample_id, str) and sample_id.isdigit():
            sample_id = int(sample_id)
        elif not isinstance(sample_id, int):
            # fallback: digits from filename
            try:
                digits = "".join(ch for ch in Path(sample["audio"]["path"]).stem if ch.isdigit())
                sample_id = int(digits) if digits else 0
            except Exception:
                sample_id = 0

        max_sample_id = max(max_sample_id, sample_id)
        text_map_en[sample_id] = sample.get("text_en", "")
        for lang in TGT_LANGS:
            text_map_tgt[lang][sample_id] = sample.get(f"text_{lang}", "")

        # Save audio as {sample_id}.wav if missing
        wav_name = f"{sample_id}.wav"
        audio_path = audio_out_dir / wav_name
        if not audio_path.exists():
            try:
                sf.write(
                    str(audio_path),
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"]
                )
            except Exception as e:
                logger.warning("audio write failed for sample_id=%s: %s", sample_id, e)

        # Common base (doc_id top-level from en.xml)
        docid = lookup_doc_id(seg_to_doc, int(sample_id))
        base_record = {
            "dataset_id": "acl_6060",
            "doc_id": docid,
            "sample_id": sample_id,
            "src_audio": f"/acl6060/audio/{SRC_LANG}/{sample_id}.wav",
            "src_ref": text_map_en.get(sample_id, ""),
            "src_lang": SRC_LANG,
            "benchmark_metadata": {
                "context": "short",
                "dataset_type": "longform",
                "subset": SPLIT
            }
        }

        for lang in TGT_LANGS:
            record = {
                **base_record,
                "tgt_ref": text_map_tgt[lang].get(sample_id, ""),
                "tgt_lang": lang,
            }
            jsonl_path = manifest_dir / f"en-{lang}.jsonl"
            mode = "x" if not jsonl_path.exists() else "a"
            try:
                with open(jsonl_path, mode, encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except FileExistsError:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Finished short records. Now appending long-form records per doc_id...")

    # ===== 1) Assign global long-form sample_id per doc_id (shared across languages) =====
    docid_to_sampleid, docid_to_filename = mapping_doc_id(en_docs, manifest_dir)

    # ===== 2) Write long-form records per language (using the shared sample_id and actual filenames) =====
    total_appended = 0
    for lang in TGT_LANGS:
        tgt_xml = xml_dir / f"{lang}.xml"
        if not tgt_xml.exists():
            logger.warning("Skip long-form for %s: missing %s", lang, tgt_xml)
            continue

        try:
            tgt_docs = doc_to_seg_pairs(tgt_xml)  # {docid: [(seg_id, tgt_text), ...]}
        except Exception as e:
            logger.error("Parse failed for %s: %s", tgt_xml, e)
            continue

        jsonl_path = manifest_dir / f"en-{lang}.jsonl"
        appended = 0
        with open(jsonl_path, "a", encoding="utf-8") as f:
            common_docids = en_docs.keys() & tgt_docs.keys()
            for docid in sorted(common_docids):
                # Join English and target texts per doc (seg order already sorted)
                en_texts  = [t for (_sid, t) in en_docs[docid]]
                tgt_texts = [t for (_sid, t) in tgt_docs[docid]]
                src_concat = "\n".join(en_texts)
                tgt_concat = "\n".join(tgt_texts)

                #sid_long = docid_to_sampleid[docid]  # shared across languages
                mapped_fname = docid_to_filename[docid]
                src_audio_path = f"/acl6060/audio/{SRC_LANG}/{mapped_fname}"

                new_record = {
                    "dataset_id": "acl_6060",
                    "doc_id": docid,
                    "sample_id": docid_to_sampleid[docid],
                    "src_audio": src_audio_path,
                    "src_ref": src_concat,
                    "tgt_ref": tgt_concat,
                    "src_lang": SRC_LANG,
                    "tgt_lang": lang,
                    "benchmark_metadata": {
                        "context": "long",
                        "dataset_type": "longform",
                        "subset": SPLIT,
                    }
                }
                f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                appended += 1
        total_appended += appended
        logger.info("Appended %d long-form records to %s (lang=%s)", appended, jsonl_path, lang)

    logger.info("Manifests generated (short + long-form).")

if __name__ == "__main__":
    main()