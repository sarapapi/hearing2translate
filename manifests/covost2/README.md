# CoVoST 2

## Overview

CoVoST 2 is derived from CommonVoice version 4.0. It consists of manually created and cleaned text translations for validated Common Voice samples in several languages. 

```bibtex
@inproceedings{WangWGP21,
  author       = {Changhan Wang and
                  Anne Wu and
                  Jiatao Gu and
                  Juan Pino},
  editor       = {Hynek Hermansky and
                  Honza Cernock{\'{y}} and
                  Luk{\'{a}}s Burget and
                  Lori Lamel and
                  Odette Scharenborg and
                  Petr Motl{\'{\i}}cek},
  title        = {CoVoST 2 and Massively Multilingual Speech Translation},
  booktitle    = {22nd Annual Conference of the International Speech Communication Association,
                  Interspeech 2021, Brno, Czechia, August 30 - September 3, 2021},
  pages        = {2247--2251},
  publisher    = {{ISCA}},
  year         = {2021},
  url          = {https://doi.org/10.21437/Interspeech.2021-2027},
  doi          = {10.21437/INTERSPEECH.2021-2027},
  timestamp    = {Tue, 11 Jun 2024 16:45:43 +0200},
  biburl       = {https://dblp.org/rec/conf/interspeech/WangWGP21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@inproceedings{wang-etal-2020-covost,
    title = "{C}o{V}o{ST}: A Diverse Multilingual Speech-To-Text Translation Corpus",
    author = "Wang, Changhan  and
      Pino, Juan  and
      Wu, Anne  and
      Gu, Jiatao",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.517",
    pages = "4197--4203",
    abstract = "Spoken language translation has recently witnessed a resurgence in popularity, thanks to the development of end-to-end models and the creation of new corpora, such as Augmented LibriSpeech and MuST-C. Existing datasets involve language pairs with English as a source language, involve very specific domains or are low resource. We introduce CoVoST, a multilingual speech-to-text translation corpus from 11 languages into English, diversified with over 11,000 speakers and over 60 accents. We describe the dataset creation methodology and provide empirical evidence of the quality of the data. We also provide initial benchmarks, including, to our knowledge, the first end-to-end many-to-one multilingual models for spoken language translation. CoVoST is released under CC0 license and free to use. We also provide additional evaluation data derived from Tatoeba under CC licenses.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
```

## Instructions

The script `generate.py` will build the jsonl files for the following language pairs:

- de-en
- es-en
- it-en
- pt-en
- zh-en

- en-es
- en-it
- en-fr
- en-de
- en-pt
- en-nl
- en-zh

The script `generate.py` will save the test set audio for these langauge pairs and generate the jsonl files in this manifest, but should only be used if the user wants to _reproduce_ these outputs. If they want to directly use them for inference, they can use the already existing jsonl files and download the test sets from a release in this github (see below).

To _reproduce_ the jsonl and audio in this manifest:

Data for three of the source langauges (it, pt, and zh) could be obtained from HuggingFace, but for en, de, es you must download CommonVoice v4 (https://commonvoice.mozilla.org/en/datasets) and put the three directories in the same directory. This directory ("--clip_dir") should contain the subdirectories `en/clips`,`es/clips`, and `de/clips`.

Once this is done:

Define the path where commonAccent will be stored:

```bash
export H2T_DATADIR=""
```

Run the python script to generate the jsonl files and save the relevant audio files. 

```bash
python3 ./generate.py --clip_dir="<CLIP_DIR>"
```

**To directly use the downloaded audio files:**

Since reproducing the audio download is time consuming, you can also directly access the test sets for running inference in these experiments:

Get the data from the relevant release by installing github cli (https://cli.github.com/) and running the following:

```bash
gh release download  data-share-covost2 --repo sarapapi/hearing2translate --pattern "covost_*"
```

As the zip folders for some languages were too large to add as a single directory, you'll need to concatenate the relevant compressed dir before unzipping:

```bash
cat covost_en.zip.part-* > covost_en.zip
unzip covost_en.zip

cat covost_de.zip.part-* > covost_de.zip
unzip covost_de.zip

cat covost_it.zip.part-* > covost_it.zip
unzip covost_it.zip

cat covost_es.zip.part-* > covost_es.zip
unzip covost_es.zip
```

You can then move the langauge specific audio dir to the expected dir for inference ()`${H2T_DATADIR}/covost2/audio/`) and proceed with inference. 

## Expected Output

After running the steps above, your directory layout will be:

```
${H2T_DATADIR}/
└─ covost2/
   └─ audio/
      └─ en/
      │  ├─ 25076709.wav
      │  ├─ 25076711.wav
      │  └─ ...
      └─ de/
      │  ├─ 19453594.wav
      │  ├─ 19453600.wav
      │  └─ ...
      └─ ...
```

If your generate.py script writes manifests, you should get JSONL files (one per language pair) under your chosen output path (e.g., ./manifests/covost2/). A jsonl entry looks like:

```json
{
  "dataset_id": "covost2",
  "sample_id": "<string>",
  "src_audio": "/covost2/audio/<src_lang>/<audio_file>",
  "src_ref": "<source raw_transcription>",
  "tgt_ref": "<target raw_transcription>",
  "src_lang": "<two-letter ISO 639-1>",
  "tgt_lang": "<two-letter ISO 639-1>",
  "benchmark_metadata": {
    "context": "short"
    }
}
```

## License

All datasets are licensed under the Creative Commons license (CC-0).