# How to use MUST-SHE 1.0

## 1. Install Dependencies

Install pandas and pathlib

```bash
pip install pandas pathlib
```

## 2. Save audio files
The dataset provides metadata and references in JSONL, but the audio must be saved locally before running inference. Currently the dataset is not available online.

- MUST-SHE_release_v1.0_open/wav/{id}.wav

```bash
export H2T_DATADIR='' #Path where the MUST-SHE dataset is stored.

python manifests/MUST-SHE/generate.py \
  -o . \
```

### Note:

- en-es.jsonl = 1164 samples
- en-it.jsonl = 1096 samples
- en-fr.jsonl = 1113 samples

All files are generated in a single execution of the script.


