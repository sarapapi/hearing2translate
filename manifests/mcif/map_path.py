from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
from typing import Dict, Iterable, List, Optional, TypedDict


class Record(TypedDict):
    iid: str
    audio_path: str


class MappingItem(TypedDict):
    iid: str
    long_path: str
    short_path: List[str]


def parse_xml(xml_path: Path) -> List[Record]:
    """Find `iid` and `audio_path` from samples with task='TRANS'."""
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    out: List[Record] = []
    for sample in root.findall(".//sample[@task='TRANS']"):
        iid = (sample.get("iid") or "").strip()
        audio_path = (sample.findtext("audio_path") or "").strip()

        if not iid:
            raise ValueError(f"Missing iid in {xml_path}")
        if not audio_path:
            raise ValueError(f"Missing audio_path for iid={iid} in {xml_path}")

        out.append(Record(iid=iid, audio_path=audio_path))
    return out


def build_short_index(short_records: Iterable[Record]) -> Dict[str, List[str]]:
    """Map iid -> list of short audio paths (split by comma)."""
    idx: Dict[str, List[str]] = {}
    for r in short_records:
        # short form contain multiple paths comma-separated
        paths = [p.strip() for p in r["audio_path"].split(",") if p.strip()]
        if not paths:
            raise ValueError(f"Empty short_path list for iid={r['iid']}")
        idx[r["iid"]] = paths
    return idx


def make_mappings(long_records: List[Record], short_index: Dict[str, List[str]]) -> List[MappingItem]:
    mappings: List[MappingItem] = []
    for r in long_records:
        iid = r["iid"]
        if iid not in short_index:
            raise KeyError(f"No matching short_path for iid={iid}")
        mappings.append(
            MappingItem(
                iid=iid,
                long_path=r["audio_path"],
                short_path=short_index[iid],
            )
        )
    return mappings


def write_jsonl(items: Iterable[MappingItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build long/short audio path mapping JSONL.")
    parser.add_argument("--root", type=Path, default=Path("xml"), help="Root folder containing XML files")
    args = parser.parse_args()

    tgt_lang = "de" # same for all tgt langs, so just pick one
    pattern = "MCIF0.2.IF.{form}.{tgt_lang}.ref.xml"

    long_file = args.root / pattern.format(form="long", tgt_lang=tgt)
    short_file = args.root / pattern.format(form="short", tgt_lang=tgt)

    long_rec = parse_xml(long_file)
    short_rec = parse_xml(short_file)

    if len(long_rec) != len(short_rec):
        raise AssertionError(
            f"Length mismatch for {tgt}: long={len(long_rec)} short={len(short_rec)}"
        )

    short_idx = build_short_index(short_rec)
    mappings = make_mappings(long_rec, short_idx)

    out_path = Path(f"id_mapping.jsonl")
    write_jsonl(mappings, out_path)


if __name__ == "__main__":
    main()