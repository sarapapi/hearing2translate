#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON on line {i} of {path}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Join input/output JSONL files into a TXT with 'src ||| hyp'.")
    ap.add_argument("--input-jsonl", required=True, help="Path to the INPUT jsonl (with src_ref & benchmark_metadata).")
    ap.add_argument("--output-jsonl", required=True, help="Path to the OUTPUT jsonl (with model outputs).")
    ap.add_argument("--txt-out", required=True, help="Destination TXT file.")
    ap.add_argument("--delimiter", default=" ||| ", help="Delimiter between source and translation (default: ' ||| ').")
    ap.add_argument("--src-key", default="src_ref", help="Key in input JSONL that holds the source sentence.")
    ap.add_argument("--hyp-key", default="output", help="Key in output JSONL that holds the translation.")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    out_txt_path = Path(args.txt_out)

    # 1) Load model outputs keyed by sample_id
    hyp_by_id = {}
    for obj in load_jsonl(output_path):
        if "sample_id" not in obj:
            print("[WARN] Output JSONL row missing 'sample_id'; skipping.", file=sys.stderr)
            continue
        sid = obj["sample_id"]
        hyp = obj.get('output').strip().replace('\n', '')
        if hyp is None:
            print(f"[WARN] sample_id={sid} has no output key; skipping.", file=sys.stderr)
            continue
        hyp_by_id[sid] = hyp

    if not hyp_by_id:
        print("[ERROR] No hypotheses loaded from output JSONL.", file=sys.stderr)
        sys.exit(1)

    # 2) Stream input jsonl in order and write paired lines when we have a match
    n_total = n_written = 0
    with open(out_txt_path, "w", encoding="utf-8") as out_f:
        for obj in load_jsonl(input_path):
            n_total += 1
            sid = obj.get("sample_id")
            if sid is None:
                print("[WARN] Input JSONL row missing 'sample_id'; skipping.", file=sys.stderr)
                continue

            src = obj.get('src_ref')
            if src is None:
                print(f"[WARN] sample_id={sid} has no src_ref key; skipping.", file=sys.stderr)
                continue

            hyp = hyp_by_id.get(sid)
            if hyp is None:
                print(f"[WARN] No hypothesis found for sample_id={sid}; skipping.", file=sys.stderr)
                continue

            out_f.write(f"{src}{args.delimiter}{hyp}\n")
            n_written += 1

    print(f"[INFO] Wrote {n_written} lines to {out_txt_path} (from {n_total} input items).")

if __name__ == "__main__":
    main()