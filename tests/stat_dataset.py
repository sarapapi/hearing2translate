import infer
import argparse
import sys
"""Script to test that all audio files in dataset manifests exist, and count their total duration.

 Usage:
 - hardlink or copy it to the root of the hearing2translate repository, where infer.py is located
 - then:

hearing2translate$ H2T_DATADIR=manifests/ python3 stat_dataset.py --in-modality speech --in-file manifests/wmt/*jsonl --model stat_dataset --out-file /dev/null 2>/dev/null
File: manifests/wmt/en-de.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
File: manifests/wmt/en-es.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
File: manifests/wmt/en-fr.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
File: manifests/wmt/en-it.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
File: manifests/wmt/en-nl.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
File: manifests/wmt/en-pt.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
File: manifests/wmt/en-zh.jsonl
- it has duration: 5794.36 seconds 1h 36m 34.36s
Total duration of all input files: 40560.52 seconds 11h 16m 0.52s

If it prints "Total duration of all input files: ...", everything is fine.
If not, run again without 2>/dev/null , read error message and investigate.
"""


parser = argparse.ArgumentParser(description="Hearing to Translate test dataset.")

infer.add_infer_args(parser, in_nargs="+")

args = parser.parse_args()
infer.MODEL_MODULES["stat_dataset"] = "tests.stat_dataset_module"
import tests.stat_dataset_module

def to_hours_minutes_seconds(duration):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60
    return f"{hours}h {minutes}m {seconds:.2f}s"

ifiles = args.in_file
kw = vars(args)
kw["continue"] = False
tot_dur = 0.0
for i in ifiles:
    kw["in_file"] = i
    print("File:", i)
    infer.infer(args)
    d = tests.stat_dataset_module.DURATION
    print(f"- it has duration: {d:.2f} seconds", to_hours_minutes_seconds(d))
    tot_dur += d
    tests.stat_dataset_module.DURATION = 0.0
print(f"Total duration of all input files: {tot_dur:.2f} seconds", to_hours_minutes_seconds(tot_dur))