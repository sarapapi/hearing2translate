# ELITR - IWSLT 2020 Non-native

## Overview
This document describes the procedure for downloading and preparing the ELITR long-form dataset. Short-form transcriptions and translations are available from the original repository, but short-form audio files are not. Therefore, only the long-form audio is used here, with text segments concatenated using spaces. The dataset is sourced from the [GitHub repository](https://github.com/ELITR/elitr-testset.git) and processed with the provided scripts.

```bibtex
@inproceedings{machacek-etal-2020-elitr,
    title = "{ELITR} Non-Native Speech Translation at {IWSLT} 2020",
    author = "Mach{\'a}{\v{c}}ek, Dominik  and
      Kratochv{\'i}l, Jon{\'a}{\v{s}}  and
      Sagar, Sangeet  and
      {\v{Z}}ilinec, Mat{\'u}{\v{s}}  and
      Bojar, Ond{\v{r}}ej  and
      Nguyen, Thai-Son  and
      Schneider, Felix  and
      Williams, Philip  and
      Yao, Yuekun",
    editor = {Federico, Marcello  and
      Waibel, Alex  and
      Knight, Kevin  and
      Nakamura, Satoshi  and
      Ney, Hermann  and
      Niehues, Jan  and
      St{\"u}ker, Sebastian  and
      Wu, Dekai  and
      Mariani, Joseph  and
      Yvon, Francois},
    booktitle = "Proceedings of the 17th International Conference on Spoken Language Translation",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.iwslt-1.25/",
    doi = "10.18653/v1/2020.iwslt-1.25",
    pages = "200--208",
    abstract = "This paper is an ELITR system submission for the non-native speech translation task at IWSLT 2020. We describe systems for offline ASR, real-time ASR, and our cascaded approach to offline SLT and real-time SLT. We select our primary candidates from a pool of pre-existing systems, develop a new end-to-end general ASR system, and a hybrid ASR trained on non-native speech. The provided small validation set prevents us from carrying out a complex validation, but we submit all the unselected candidates for contrastive evaluation on the test set."
}
```

## Requirements
`pydub`

## Instructions

1. Download the audio files.
```
# Set the root directory
export H2T_DATADIR='/path/to/data'

# Run the processing script
./manifests/elitr/fetch_elitr.sh
```

2. Run the processing script.
```
python manifests/elitr/generate.py
```


## Expected Output
The process generates 48 audio files and 7 En-to-X manifest files. Target translations are provided for German.

.
├── elitr-long
│   └── audio/
│       └── en/
│           ├── 0.wav
│           ├── 1.wav
│           └── ...
│
├── manifests
│   └── elitr-long
│       ├── en-de.jsonl
│       ├── en-fr.jsonl
│       ├── ...



## License
CC BY
