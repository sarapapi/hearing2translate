# MCIF Short

## Overview
This repository provides instructions for downloading and preparing the MCIF v1.0 Short dataset.
The dataset is released as part of the [GitHub release](https://github.com/sarapapi/hearing2translate/releases/tag/data-share-mcif-short) and can be directly used for model inference.

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

⚠️ Note: The current manifests do not include `src_ref` or `tgt_ref` keys due to segmentation issues.
Before applying evaluation metrics, the model hypotheses, transcripts, and reference translations must first be segmented and aligned using a segmenter.

## Expected Output
The dataset contains 670 audio files, and manifests are provided for three language pairs: 

- en-de
- en-it
- en-zh


```
${H2T_DATADIR}/
├── mcif-short
│   └── audio/
│       └── en/
│           ├── 0.wav
│           ├── 1.wav
│           └── ...
│
├── manifests
│   └── mcif-short
│       ├── en-de.jsonl
│       ├── en-it.jsonl
│       ├── en-zh.jsonl

```

## License
CC BY 4.0