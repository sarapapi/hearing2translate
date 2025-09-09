# ACL60/60 Long

## Overview
We describe the process for downloading and preparing the ACL 60/60 dataset. The process involves downloading the main dataset from [git release](https://github.com/sarapapi/hearing2translate/releases/tag/data-share-acl6060) and running a processing script.

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


## Instructions

1. Download and extract the dataset using fetch_data.sh.
```
# Set the root directory
export H2T_DATADIR='/path/to/data'

# Download and extract
./fetch_data.sh
```

2.	The manifests are already provided. You do not need to regenerate them.
    
## Expected Output
The process produces 5 audio files and 7 En-to-X manifest files. Target translations are available for German, French, Portuguese, and Chinese.

.
├── acl6060-long
│   └── audio/
│       └── en/
│           ├── 416.wav
│           ├── 417.wav
│           └── ...
│
├── manifests
│   └── acl6060-long
│       ├── en-de.jsonl
│       ├── en-fr.jsonl
│       ├── ...



## License
CC BY 4.0