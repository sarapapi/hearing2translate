# FLEURS

## Overview

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech), is a benchmark dataset for speech research. The dataset is an n-way parallel speech dataset that includes 102 languages and is based on the machine translation FLoRes-101 benchmark. It contains approximately 12 hours of speech per language. The FLEURS benchmark enables the evaluation of various speech tasks, including Automatic Speech Recognition (ASR), Speech Language Identification (Speech LangID), Translation, and Retrieval.

```bibtex
@article{fleurs2022arxiv,
  title = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  url = {https://arxiv.org/abs/2205.12446},
  year = {2022}
}
```

## Instructions

Define the path where **FLEURS** will be stored:

```bash
export H2T_DATADIR=""
```

Run the Python script to generate the processed data:

```bash
python generate.py
```

## Expected Output

After running the steps above, your directory layout will be:

```
${H2T_DATADIR}/
└─ fleurs/
   └─ audio/
      └─ en/
      │  ├─ 14738234113419638776.wav
      │  ├─ 17498257810809617374.wav
      │  └─ ...
      └─ de/
      │  ├─ 2835934118517986318.wav
      │  ├─ 14644395854086367094.wav
      │  └─ ...
      └─ ...
```

If your generate.py script writes manifests, you should get JSONL files (one per language pair) under your chosen output path (e.g., ./manifests/Fleurs/). A jsonl entry looks like:


```json
{
  "dataset_id": "fleurs",
  "sample_id": "<string>",
  "src_audio": "/fleurs/audio/<src_lang>/<audio file>",
  "src_ref": "<source raw_transcription>",
  "tgt_ref": "<target raw_transcription>",
  "src_lang": "<two-letter ISO 639-1>",
  "tgt_lang": "<two-letter ISO 639-1>",
  "benchmark_metadata": {"gender": "0|1"}
}
```

## License

All datasets are licensed under the Creative Commons license (CC-BY).
