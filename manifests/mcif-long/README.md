# MCIF Long

## Overview
This repository provides instructions for downloading and preparing the MCIF v1.0 dataset.
The dataset is released as part of the [GitHub release](https://github.com/sarapapi/hearing2translate/releases/tag/data-share-mcif) and can be directly used for model inference.

```bibtex
@misc{papi2025mcifmultimodalcrosslingualinstructionfollowing,
      title={MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks}, 
      author={Sara Papi and Maike Züfle and Marco Gaido and Beatrice Savoldi and Danni Liu and Ioannis Douros and Luisa Bentivogli and Jan Niehues},
      year={2025},
      eprint={2507.19634},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19634}, 
}
```

## Instructions

1.	Download the dataset

Download the release package and place the files under your desired root directory.
```
# Set the root directory
export H2T_DATADIR='/path/to/data'
```

2.	Use provided manifests

The dataset already includes pre-generated manifests.
There is no need to regenerate them unless you modify the data.

## Expected Output
The dataset contains 21 audio files, and manifests are provided for three language pairs: 

- en-de
- en-it
- en-zh


```
${H2T_DATADIR}/
├── mcif-long
│   └── audio/
│       └── en/
│           ├── 0.wav
│           ├── 1.wav
│           └── ...
│
│       └── long_texts/
│           ├── crgYiwKDfX.en
│           ├── ...
│
│       └── xml/
│           ├── MCIF0.2.IF.long.de.ref.xml
│           ├── ...
│
├── manifests
│   └── mcif-long
│       ├── en-de.jsonl
│       ├── en-it.jsonl
│       ├── en-zh.jsonl

```

## License
CC BY 4.0