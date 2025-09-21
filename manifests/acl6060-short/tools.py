import re
import html
import json
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

# -------------------- Paths --------------------
BASE_DIR = Path(__file__).resolve().parent
XML_FILE = BASE_DIR.parent / "acl6060-long" / "en.xml"

# -------------------- Precompiled regex --------------------
_TALK_RE = re.compile(r"<talkid>\s*(.*?)\s*</talkid>", re.IGNORECASE | re.DOTALL)
_SEG_RE  = re.compile(r'<seg\b[^>]*\bid\s*=\s*"([^"]+)"[^>]*>(.*?)</seg>',
                      re.IGNORECASE | re.DOTALL)
_TAG_RE  = re.compile(r"<[^>]+>")  # strip inner tags


# -------------------- Utils --------------------
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


# -------------------- Utils --------------------
def _clean_text(s: str) -> str:
    s = _TAG_RE.sub("", s)
    s = html.unescape(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _iter_talk_blocks(text: str) -> Iterable[Tuple[str, str]]:
    talks = list(_TALK_RE.finditer(text))
    for i, tm in enumerate(talks):
        talkid = tm.group(1).strip()
        start = tm.end()
        end = talks[i + 1].start() if i + 1 < len(talks) else len(text)
        yield talkid, text[start:end]


def _extract_seg_texts(block: str) -> List[str]:
    seg_texts: List[str] = []
    for m in _SEG_RE.finditer(block):
        seg_inner = m.group(2)
        seg_text = _clean_text(seg_inner)
        if seg_text:
            seg_texts.append(seg_text)
    return seg_texts


# -------------------- Builders --------------------
def build_talkid_to_allsegs(xml_path: str | Path = XML_FILE) -> Dict[str, str]:
    """
    Create { talkid: "seg1 seg2 seg3 ..." }
    """
    xml_path = Path(xml_path)
    text = xml_path.read_text(encoding="utf-8", errors="replace")

    mapping: Dict[str, str] = {}
    for talkid, block in _iter_talk_blocks(text):
        seg_texts = _extract_seg_texts(block)
        mapping[talkid] = " ".join(seg_texts) if seg_texts else ""
    return mapping


def build_segtext_to_talkid(xml_path: str | Path = XML_FILE) -> Dict[str, str]:
    """
    Create { seg_text: talkid } 
    """
    xml_path = Path(xml_path)
    text = xml_path.read_text(encoding="utf-8", errors="replace")

    mapping: Dict[str, str] = {}
    for talkid, block in _iter_talk_blocks(text):
        for seg_text in _extract_seg_texts(block):
            mapping[seg_text] = talkid
    return mapping


# -------------------- CLI --------------------
if __name__ == "__main__":
    data = build_talkid_to_allsegs(XML_FILE)

    out_file = BASE_DIR.parent / "acl6060-long" / "long.src.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved: {out_file} (talks={len(data)})")