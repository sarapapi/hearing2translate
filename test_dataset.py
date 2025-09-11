import infer
import argparse
import sys
"""Script to test that all files in dataset manifests exist

 Usage:
 - in hearing2translate directory, where infer.py is located

H2T_DATADIR=manifests/ python3 test_dataset.py  --in-modality speech --in-file manifests/fleurs/*jsonl 2>/dev/null

If it prints "Success!", everything is fine.
Otherwise it crashes on some error. Then read the standard error message and investigate.
"""


parser = argparse.ArgumentParser(description="Hearing to Translate test dataset.")

# TODO: code duplication with infer.py is not nice
parser.add_argument("--model", choices=["test_dataset"], default="test_dataset",
                    help="Model to be used for inference")
parser.add_argument("--in-modality", choices=["speech", "text"], required=True,
                    help="Input modality used for inference")
parser.add_argument("--in-file", required=True, help="Input JSONL files path", nargs="+")
parser.add_argument("--out-file", required=False, help="Output JSONL file path. If not set: stdout.", default=None)
parser.add_argument("--transcript-file",
                    help="Optional JSONL with transcripts for text modality")
parser.add_argument("--asr", default=False, action="store_true",
                        help="If set, the speech model is used as ASR for the src lang. Tgt language is ignored.")

args = parser.parse_args()
infer.MODEL_MODULES["test_dataset"] = "tests.test_dataset_module"

ifiles = args.in_file
kw = vars(args)
for i in ifiles:
    kw["in_file"] = i
    print("Testing file:", i, file=sys.stderr)
    infer.infer(args)

print("Success! All audio files in the input files exist.", file=sys.stderr)