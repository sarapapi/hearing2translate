import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--src",default="en",help="source language code")
parser.add_argument("--tgt",default="e",help="one target language code")
parser.add_argument("--inputs",default=[],help="input files", nargs='+')
parser.add_argument("--long",default=False, action="store_true",help="Whether the processing should be long-form.")

args = parser.parse_args()

if args.inputs == []:
    print("no input file given, terminating",file=sys.stderr)
    sys.exit(1)

if args.tgt == "en":
    if args.src != "en":
        task = "translate"
    else:
        task = "transcribe"
elif args.tgt != args.src:
    print("wrong src or tgt. whisper can only transcribe or translate to en",file=sys.stderr)
    sys.exit(1)
else:
    task = "transcribe"

if args.src == "en" and args.tgt != "en":
    print("whisper can only transcribe or translate to en",file=sys.stderr)
    sys.exit(1)




# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")


transcriptions = pipe(args.inputs, generate_kwargs={"language": args.src, "task": task}, 
                        # because it doesn't work without
                      chunk_length_s=30, stride_length_s=(5,5) if args.long else (0,0), return_timestamps="word")
print(transcriptions)