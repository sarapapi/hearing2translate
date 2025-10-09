# Libristutter

## Overview

Libristutter is a dataset designed to simulate stuttered speech. It is derived from the publicly available LibriSpeech corpus by artificially introducing various types of disfluencies, including sound, word, and phrase repetitions, as well as prolongations and interjections. The dataset comprises roughly 20 hours of audio, divided into 4,26k training utterances and 474 test utterances.

```bibtex
@ARTICLE{9528931,
  author={Kourkounakis, Tedd and Hajavi, Amirhossein and Etemad, Ali},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={FluentNet: End-to-End Detection of Stuttered Speech Disfluencies With Deep Learning}, 
  year={2021},
  volume={29},
  number={},
  pages={2986-2999},
  keywords={Speech processing;Deep learning;Training;Benchmark testing;Tools;Speaker recognition;Residual neural networks;Attention;disfluency;deep learning;BLSTM;speech;stutter;squeeze-and-excitation},
  doi={10.1109/TASLP.2021.3110146}}
```

## Instructions

Define the path where **Libristutter** will be stored:

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
└─ libristutter/
   └─ audio/
      └─ en/
        ├─6000-55211-0004.flac
        ├─ 2518-154825-0004.flac
        ├─ 4018-107312-0003.flac
        ├─ 625-132118-0031.flac
        └─ ...
      
```

If your generate.py script writes manifests, you should get JSONL files (one per language pair) under your chosen output path (e.g., ./manifests/libristutter/). A jsonl entry looks like:

```json
{
  "dataset_id": "libristutter",
  "sample_id": "<string>",
  "src_audio": "/libristutter/audio/en/<audio file>",
  "src_ref": "<source raw_transcription>",
  "tgt_ref": "null",
  "src_lang": "en",
  "tgt_lang": "<two-letter ISO 639-1 >",
  "benchmark_metadata": {"has_stutter":"True|False", "stutter_pos":[]}
}
```

## License

All datasets are licensed under the Creative Commons Non Commercial license (CC BY-NC 4.0).
