import infer
import argparse
import sys
"""Script to test that all files in dataset manifests exist

 Usage:
 - hardlink or copy this file to the root of the hearing2translate repository, where infer.py is located
 - then:

H2T_DATADIR=manifests/ python3 test_dataset.py  --in-modality speech --in-file manifests/fleurs/*jsonl

If it prints "Success!", everything is fine.
Otherwise it crashes on some error. Then read the standard error message and investigate.
"""


parser = argparse.ArgumentParser(description="Hearing to Translate test dataset.")

infer.add_infer_args(parser, is_test=True)

args = parser.parse_args()
import tests.test_dataset_module as t
infer.setup_model = lambda model_name, modality: (t.load_model(), t.generate)

ifiles = args.in_file
kw = vars(args)
kw["continue"] = False
for i in ifiles:
    kw["in_file"] = i
    print("Testing file:", i, file=sys.stderr)
    infer.infer(args)

print("Success! All audio files in the input files exist.", file=sys.stderr)
