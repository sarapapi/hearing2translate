# How to use ACL 60/60

## 1. Install Dependencies

We don’t need `torchcodec`. Instead, use `datasets<4.0` with built-in audio decoding:

```bash
pip install "datasets[audio]<4.0" soundfile
```

## 2. Save audio files
The dataset provides metadata and references in JSONL, but the audio must be saved locally before running inference. Since the **source language is English**, you just need to extract audio separately for the available splits (`dev` and `eval`). We store audio files per split:

- dev  → audio/dev/{sample_id}.wav
- eval → audio/eval/{sample_id}.wav

```bash
# Dev split
python manifests/acl6060/save_audio.py \
  --jsonl manifests/acl6060/en-de.jsonl \
  --split dev

# Eval split
python manifests/acl6060/save_audio.py \
  --jsonl manifests/acl6060/en-de.jsonl \
  --split eval
```

### Note:

- dev split = 468 samples
- eval split = 416 samples
- Total = 884 samples in the manifest.

You need to run the script for both splits to cover all audio.


## 3. Run inference
Once audio is prepared, run inference with a model supported by `infer.py`.

```bash
python infer.py \
  --model {model_name} \
  --in-modality speech \
  --in-file manifests/acl6060/en-de.jsonl \
  --out-file outputs/en-de_preds.jsonl
```

## Structure
```
h2t/
├─ infer.py
├─ manifests/
│  └─ acl6060/
│     ├─ en-de.jsonl
│     ├─ save_audio.py
│     └─ audio/
│        ├─ dev/
│        │  ├─ 0.wav
│        │  └─ ...
│        └─ eval/
│           ├─ 0.wav
│           └─ ...
└─ outputs/
   └─ en-de_preds.jsonl
```
