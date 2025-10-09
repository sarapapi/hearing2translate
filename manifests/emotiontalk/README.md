# EmotionTalk

## Overview

[**EmotionTalk**](https://huggingface.co/datasets/BAAI/Emotiontalk) is an interactive Chinese multimodal emotion dataset built by the Beijing Academy of Artificial Intelligence. It contains **23.6 hours** of conversations (19,250 utterances) across audio, video, and text modalities. Seven discrete emotions are annotated at the utterance level (happy, surprise, sad, disgust, anger, fear, neutral), along with multi-dimensional sentiment/speech descriptions.

Our pipeline targets only the **test split**, which consists of recordings whose identifiers begin with `G00003` and `G00015`. Each utterance is processed independently, with Mandarin speech as the source (`zh`) and no target reference.

```bibtex
@article{sun2025emotiontalk,
  title={EmotionTalk: An Interactive Chinese Multimodal Emotion Dataset With Rich Annotations},
  author={Sun, Haoqin and Wang, Xuechen and Zhao, Jinghua and Zhao, Shiwan and Zhou, Jiaming and Wang, Hui and He, Jiabei and Kong, Aobo and Yang, Xi and Wang, Yequan and others},
  journal={arXiv preprint arXiv:2505.23018},
  year={2025}
}
```

## Instructions

1. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
2. Define the path where **EmotionTalk** will be stored:

  ```bash
  export H2T_DATADIR=""
  ```
3. Run the script to download and prepare the dataset:

  ```bash
  python generate.py
  ```

> Notes:
> - Only the EmotionTalk test utterances (IDs starting with `G00003` or `G00015`) are downloaded and expanded.

## Expected Outputs

After execution, the data directory will resemble:

```
${H2T_DATADIR}/
└─ emotiontalk/
   ├── audio/
   │   └── zh/
   │       ├── G00003_XXXX.wav
   │       ├── G00015_XXXX.wav
   │       └── ...
   ├── raw/
   │   └── Audio.tar
   └── zh-en.jsonl
```

Each JSONL record looks like:

```json
{
  "dataset_id": "emotiontalk",
  "sample_id": "<sample_id>",
  "src_audio": "emotiontalk/audio/zh/<sample_id>.wav",
  "src_ref": "<source raw_transcription>",
  "tgt_ref": null,
  "src_lang": "zh",
  "tgt_lang": "en",
  "benchmark_metadata": {
    "context": "short",
    "emotion": "<emotion>",
    "dataset_type": "emotion"
  }
}
```

## License

EmotionTalk is distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License (CC BY-NC-SA 4.0)**. Review the license before using the dataset.【https://huggingface.co/datasets/BAAI/Emotiontalk】


