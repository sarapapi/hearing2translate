import os
import json
import argparse
import pandas as pd
from pathlib import Path


# --- Configuration ---
# Define the language pairs you want to process.
# The format is 'source_language-target_language'.
# These should be two-letter ISO 639-1 language codes.
LANGUAGE_PAIRS = [
    "en-fr",
    "en-it",
    "en-es",
]


def load_must_she_dataset(lang_pair, monolingual_path):
    tgt_lang = lang_pair.split("-")[-1]
    data_filename = Path("MONOLINGUAL.open." + tgt_lang + ".tsv")
    full_path = dataset_path / data_filename
    df = pd.read_csv(full_path, delimiter='\t')
    return df, full_path
    

def process_must_she_dataset(df, lang_pair, out_path):
    jsonl_filename = out_path + '/' +  lang_pair + ".jsonl"
    src_lang, tgt_lang = lang_pair.split("-")
    with  open(jsonl_filename, "w", encoding="utf-8") as f_json:
        for _,example in df.iterrows():
            # Construct the JSON record
            record = {
                "dataset_id": "MUST-SHE",
                "sample_id": example['ID'],
                "src_audio": 'wav/' + example['ID'] + '.wav',
                "src_ref": example['SRC'],
                "tgt_ref": example['REF'],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "benchmark_metadata": {
                    "gender": example['GENDER'], 
                    "talk": example['TALK'], 
                    "speaker": example['SPEAKER'], 
                    "wrong-ref": example['WRONG-REF'], 
                    "category": example['CATEGORY'],
                    "comment": example['COMMENT'],
                    "free-ref": example['FREE-REF'],
                    "context": "short"
                }
            }

            f_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create jsonl format files for the MUST-SHE dataset")
    ap.add_argument("-o", "--out_path", required=True, help="Path to JSONL manifest files")
    args = ap.parse_args()
    dataset_path = Path(os.getenv("H2T_DATADIR"))
    for lang_pair in LANGUAGE_PAIRS:
        df, monolingual_path = load_must_she_dataset(lang_pair, dataset_path)
        process_must_she_dataset(df, lang_pair, args.out_path)
