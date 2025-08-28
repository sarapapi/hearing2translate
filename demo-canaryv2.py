import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--src",default="en","source language code")
parser.add_argument("--tgt",default="de",help="one target language code")
parser.add_argument("--inputs",default=[],help="input files")
parser.add_argument("--long",default=False, action="store_true",help="Whether the processing should be long-form.")

args = parser.parse_args()

if args.inputs == []:
    print("no input file given, terminating",file=sys.stderr)
    sys.exit(1)

import nemo.collections.asr as nemo_asr
#asr_model = nemo_asr.models.ASRModel.from_pretrained(model_path="canaryv2/canary-1b-v2/")
asr_model = nemo_asr.models.ASRModel.restore_from(restore_path="canaryv2/canary-1b-v2/canary-1b-v2.nemo")

transcriptions = asr_model.transcribe(args.inputs, source_lang=args.src, target_lang=args.tgt)

print(transcriptions)
