# ACL60/60 Short

## Overview
This document outlines the procedure for downloading and preparing the ACL 60/60 short-form dataset. The dataset is sourced from Hugging Face and then processed using a provided script.

```bibtex
@inproceedings{salesky-etal-2023-evaluating,
    title = "Evaluating Multilingual Speech Translation under Realistic Conditions with Resegmentation and Terminology",
    author = "Salesky, Elizabeth  and
      Darwish, Kareem  and
      Al-Badrashiny, Mohamed  and
      Diab, Mona  and
      Niehues, Jan",
    editor = "Salesky, Elizabeth  and
      Federico, Marcello  and
      Carpuat, Marine",
    booktitle = "Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.iwslt-1.2/",
    doi = "10.18653/v1/2023.iwslt-1.2",
    pages = "62--78",
    abstract = "We present the ACL 60/60 evaluation sets for multilingual translation of ACL 2022 technical presentations into 10 target languages. This dataset enables further research into multilingual speech translation under realistic recording conditions with unsegmented audio and domain-specific terminology, applying NLP tools to text and speech in the technical domain, and evaluating and improving model robustness to diverse speaker demographics."
}
```

## Requirements
`dataset==3.6`

## Instructions

1. Generate the processed data.
```
# Set the root directory
export H2T_DATADIR='/path/to/data'

# Run the processing script
python manifests/acl6060-short/generate.py
```

### NOTE!
- On the first run, the manifest and audio files will be created.
- On subsequent runs, new records will be appended to the existing manifest.

## Expected Output
The process generates 416 audio files and 7 En-to-X manifest files. Target translations are provided for German, French, Portuguese, and Chinese.

```
.
├── acl6060-short
│   └── audio/
│       └── en/
│           ├── 0.wav
│           ├── 1.wav
│           └── ...
│
├── manifests
│   └── acl6060-short
│       ├── en-de.jsonl
│       ├── en-fr.jsonl
│       ├── ...

```

## License
CC BY 4.0