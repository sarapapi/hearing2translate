# CommonAccent

The script `generate.py` is used to access and download the relevant audio files for CommonAccent from CommonVoice v11. It also creates the jsonl files to be used as input for inference for each language pair. The langauge pairs from this benchmark are:

- de-en
- es-en
- it-en

- en-es
- en-it
- en-fr
- en-de
- en-pt
- en-nl
- en-zh

The `tgt_ref` field will always be "null" as the dataset has no reference translations. We can therefore generate translations in all possible target langauges. 

# How to use

## 1. Install dependencies

```bash 
pip install -r requirements.txt
```

## 2. Save Audio files

The script `generate.py` script will get the audio files from CommonVoice v11 on HuggingFace, save them, and save the input jsonl files to the current directory. 

```bash
python3 ./generate.py
```