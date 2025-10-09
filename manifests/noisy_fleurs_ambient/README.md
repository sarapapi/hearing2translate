# NOSIY FLEURS AMBIENT

## Overview

Noisy-Fleurs-Ambient is a derivative of the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset.

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech), is a benchmark dataset for speech research. The dataset is an n-way parallel speech dataset that includes 102 languages and is based on the machine translation FLoRes-101 benchmark. It contains approximately 12 hours of speech per language. 

We add two types of realistic noise (babble and ambient) sourced from the [MUSAN](https://www.openslr.org/17) corpus to simulate challenging acoustic conditions using the method of Anwar et. al (2023) in MuAViC.

- Babble noise consists of overlapping human speech (multiple speakers talking simultaneously).
- Ambient noise includes a balanced mix of environmental sounds (e.g., car, siren, phone) and music samples from MUSAN.


```bibtex
@article{fleurs2022arxiv,
  title = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  url = {https://arxiv.org/abs/2205.12446},
  year = {2022}
}
```

```bibtex
@inproceedings{anwar23_interspeech,
  title     = {MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation},
  author    = {Mohamed Anwar and Bowen Shi and Vedanuj Goswami and Wei-Ning Hsu and Juan Pino and Changhan Wang},
  year      = {2023},
  booktitle = {Interspeech 2023},
  pages     = {4064--4068},
  doi       = {10.21437/Interspeech.2023-2279},
  issn      = {2958-1796},
}
```

```bibtex
@misc{musan2015,
  author = {David Snyder and Guoguo Chen and Daniel Povey},
  title = {{MUSAN}: {A} {M}usic, {S}peech, and {N}oise {C}orpus},
  year = {2015},
  eprint = {1510.08484},
  note = {arXiv:1510.08484v1}
}
```

## Instructions

Define the path where **NOISY FLEURS AMBIENT** will be stored:

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
└─ noisy_fleurs_ambient/
    └─ audio
      └─ en/
      │  ├─ 14738234113419638776_ambient.wav
      │  ├─ 17498257810809617374_ambient.wav
      │  └─ ...
      └─ de/
      │  ├─ 2835934118517986318_ambient.wav
      │  ├─ 14644395854086367094_ambient.wav
      │  └─ ...
      └─ ...
```

The manifest jsonl files will be stored per language. A jsonl entry looks like:


```json
{
  "dataset_id": "noisy_fleurs_ambient",
  "sample_id": "<string>",
  "src_audio": "noisy_fleurs_ambient_wavs/<src_lang>/<audio_file>",
  "src_ref": "<source_raw_transcription>",
  "tgt_ref": "<target_raw_transcription>",
  "src_lang": "<two-letter ISO 639-1 code>",
  "tgt_lang": "<two-letter ISO 639-1 code>"
  "benchmark_metadata": {"gender": "0|1"}
}
```

## License

All datasets are licensed under the Creative Commons license CC BY-NC 4.0.

## Adding noise manually

1. **Download** the [MUSAN dataset](https://www.openslr.org/17) and [FLEURS dataset](../fleurs/).  
2. **Edit** the paths in the script [`generate_noisy_fleurs.sh`](./generate_noisy_fleurs.sh) to match your environment.  
   - You can also specify the desired noise type: `babble` or `ambient`.  
3. **Run** the script to add noise to FLEURS:
   ```bash
   bash generate_noisy_fleurs.sh