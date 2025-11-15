import argparse
import csv
import collections
import math

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", type=str, nargs="+", required=True, help="Input CSV files")
args.add_argument("-o", "--output", type=str, required=True, help="Output CSV file")
args = args.parse_args()

LANG_ORDER = [
    "en-es", "en-fr", "en-pt", "en-it", "en-de", "en-nl", "en-zh",
    "es-en", "fr-en", "pt-en", "it-en", "de-en"
]

METRICS_ORDER = [
    "LinguaPy",
    "metricx_qe_score",
    "QEMetricX_24-Strict-linguapy",
    "xcomet_qe_score",
    "XCOMET-QE-Strict-linguapy",
]

SYSTEM_ORDER = [
    # --- SFM ---
    "whisper",
    "seamlessm4t",
    "canary-v2",
    "owsm4.0-ctc",

    # --- Cascade ---
    "aya_whisper", #missing
    "gemma_whisper",
    "tower_whisper",

    "aya_seamlessm4t",#missing
    "gemma_seamlessm4t",
    "tower_seamlessm4t",

    "aya_canary-v2",
    "gemma_canary-v2",
    "tower_canary-v2",

    "aya_owsm4.0-ctc", #missing
    "gemma_owsm4.0-ctc",
    "tower_owsm4.0-ctc",

    # --- SpeechLLM ---
    "desta2-8b",
    "qwen2audio-7b",
    "phi4multimodal",
    "voxtral-small-24b",
    "spirelm",
]

def safe_float(x):
    try:
        x = x.strip()
        return float(x) if x != "" else math.nan
    except Exception:
        return math.nan

data = collections.defaultdict(lambda: collections.defaultdict(dict))
metrics = None

for fname in args.input:
    langs = fname.split("/")[-1].split(".")[0].split("_", 1)[1].replace("_", "-")
    print(f"Processing {langs}: {fname}")
    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system = row.pop("system")
            # Store metrics with safe float conversion
            data[langs][system] = {k.strip(): safe_float(v) for k, v in row.items()}
            if metrics is None:
                metrics = [k.strip() for k in row.keys()]

langs = [l for l in LANG_ORDER if l in data]

if not langs:
    raise ValueError("No matching language pairs found in input files.")

#all_systems = sorted({s for l in langs for s in data[l].keys()})
#all_systems = sorted({s for l in langs for s in data[l].keys()})
#available_systems = {s for l in langs for s in data[l].keys()}
#all_systems = [s for s in SYSTEM_ORDER if s in available_systems]
all_systems = [s for s in SYSTEM_ORDER]
#all_systems += sorted(available_systems - set(all_systems))

with open(args.output, "w", newline="") as f:
    def printcsv(*args):
        print(*args, sep=",", file=f)

    #header
    printcsv(
        "system",
        *[
            x
            for metric in metrics
            for x in [metric] + ["" for _ in langs[:-1]]
        ]
    )

    #lang code
    printcsv(
        "",
        *[
            lang
            for _ in metrics
            for lang in langs
        ]
    )

    for system in all_systems:
        row = []
        for metric in metrics:
            for lang in langs:
                row.append(data[lang].get(system, {}).get(metric, ""))
        printcsv(system, *row)

print(f"Combined CSV written to {args.output}")