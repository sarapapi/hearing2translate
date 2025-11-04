"""
Combines multiple CSV files together into a single file (grouped by metrics) that can be used in GSheet.

Example usage: python3 analysis/wmt/combine_csv.py -i analysis/wmt/*.csv -o /home/vilda/Downloads/wmt_combined.csv
"""

import argparse
import csv
import collections

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", type=str, nargs="+", required=True, help="Input CSV files")
args.add_argument("-o", "--output", type=str, required=True, help="Output CSV file")
args = args.parse_args()

data = collections.defaultdict(lambda: collections.defaultdict(dict))
metrics = None

for fname in args.input:
    langs = fname.split("/")[-1].split(".")[0].split("_", 1)[1].replace("_", "-")
    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system = row.pop("system")
            data[langs][system] = {k: float(v) for k, v in row.items()}
            
            if metrics is None:
                metrics = list(row.keys())

langs = list(data.keys())


with open(args.output, "w") as f:
    def printcsv(*args):
        print(
            *args,
            sep=",",
            file=f,
            end="\n",
        )
    
    printcsv(
        "system",
        *[
            x
            for metric in metrics
            for x in [metric] + ["" for _lang in langs[:-1]]
        ]
    )
    printcsv(
        "",
        *[
            lang
            for _ in metrics
            for lang in langs
        ]
    )
    for system in data[langs[0]].keys():
        printcsv(
            system,
            *[
                data[lang][system][metric]
                for metric in metrics
                for lang in langs
            ]
        )