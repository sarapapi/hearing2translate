# CS-Dialogue

## Overview

**CS-Dialogue** is a dataset of spontaneous Mandarin–English code-switching speech. It contains **104 hours** of audio from **200 speakers**, recorded across **100 dialogues** with a total of **38,917 utterances**. All speakers are native Chinese and fluent in English, with diverse age and regional backgrounds. The recordings cover **7 topics**: personal, entertainment, technology, education, job, philosophy, and sports. The dataset is divided into **train**, **development**, and **test** splits.

The linguistic composition of the dialogue progressed systematically: initially in Mandarin Chinese, followed by a period of code-switching between Chinese and English, and concluding with exclusive use of English. Each topic segment was designed to last approximately 20 minutes, with a target allocation of 8 minutes for Chinese, 6 minutes for code-switching, and 6 minutes for English. Since we are only interested in the code-switching aspect, **we only use the portion of the test split that is code-switching** (which mainly consists of **Chinese with embedded English**).

```bibtex
@article{zhou2025cs,
  title={CS-Dialogue: A 104-Hour Dataset of Spontaneous Mandarin-English Code-Switching Dialogues for Speech Recognition},
  author={Zhou, Jiaming and Guo, Yujie and Zhao, Shiwan and Sun, Haoqin and Wang, Hui and He, Jiabei and Kong, Aobo and Wang, Shiyao and Yang, Xi and Wang, Yequan and others},
  journal={arXiv preprint arXiv:2502.18913},
  year={2025}
}
```

## Instructions

1. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
2. Define the path where **CS-Dialogue** will be stored:

  ```bash
  export H2T_DATADIR=""
  ```
3. Run the script to download and prepare the dataset:

  ```bash
  python generate.py
  ```
> Note: Extracting .tar files may take some time.
> The script prepares manifests without reference translations.

## Expected Outputs

After running the above steps, your directory should look like:

```
${H2T_DATADIR}/
└─ cs-dialogue/
    ├── audio
    │   └── zh
    │       ├── ZH-CN_U0001_S0_101.wav
    │       ├── ZH-CN_U0001_S0_103.wav
    │       ├── ZH-CN_U0001_S0_105.wav
    │       └── ...
    ├── raw
    │    └── ...
    └── zh-en.jsonl
```

Manifests will be generated under your chosen output path (e.g., `./manifests/cs-dialogue/`).

Each entry in the JSONL manifest looks like:


```json
{
  "dataset_id": "cs-dialogue",
  "sample_id": "<dialogue_id>_<utterance_idx>",
  "src_audio": "/cs-dialogue/audio/zh/<dialogue_id>_<utterance_idx>.wav",
  "src_ref": "<source raw_transcription>",
  "tgt_ref": null,
  "src_lang": "zh",
  "tgt_lang": "en",
  "benchmark_metadata": {
    "cs_lang": ["en", "zh"],
    "context": "short",
    "dataset_type": "code_switch"
  }
}

```


## License

**CS-Dialogue** is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License (CC BY-NC-SA 4.0)**.
