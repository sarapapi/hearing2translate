# WinoST

## Overview

WinoST is a challenge set designed to evaluate gender bias in speech translation. WinoST is the spoken counterpart of WinoMT and contains 3,888 English audio files recorded by a female American speaker in WAV format (48 kHz, 16-bit). It provides stereotypical and anti-stereotypical examples, where professions and pronouns must be correctly matched for gender in translation. The benchmark enables systematic evaluation of gender bias by measuring how accurately speech translation systems preserve gendered references across languages, revealing persistent stereotypical errors (e.g., always translating nurse as female or developer as male).

```bibtex
@inproceedings{costa-jussa-etal-2022-evaluating,
    title = "Evaluating Gender Bias in Speech Translation",
    author = "Costa-juss{\`a}, Marta R.  and
      Basta, Christine  and
      G{\'a}llego, Gerard I.",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.230/",
    pages = "2141--2147",
    }

@inproceedings{stanovsky-etal-2019-evaluating,
    title = "Evaluating Gender Bias in Machine Translation",
    author = "Stanovsky, Gabriel  and
      Smith, Noah A.  and
      Zettlemoyer, Luke",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\`a}rquez, Llu{\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1164/",
    doi = "10.18653/v1/P19-1164",
    pages = "1679--1684",
}
```


## Instructions

Define the path where **WinoST** will be stored:

```bash
export H2T_DATADIR=""
mkdir -p "${H2T_DATADIR}/winoST/audio/en"
```

Download and unzip the **WinoST** dataset from Zenodo:

```bash
wget -q "https://zenodo.org/records/4139080/files/WinoST.zip?download=1" -O "${H2T_DATADIR}/winoST/WinoST.zip"

# 2) Unzip to a temp dir
tmpdir="${H2T_DATADIR}/winoST/.tmp_extract"
rm -rf "$tmpdir"
mkdir -p "$tmpdir"
unzip -qq -d "$tmpdir" "${H2T_DATADIR}/winoST/WinoST.zip"
rm -rf "${tmpdir}/__MACOSX"
````

Decompress the audio files (`.wav.gz`):

```bash
find "$tmpdir" -type f -name "*.wav.gz" -print0 | xargs -0 -I{} gunzip "{}"
```

Move all wav files into the main folder:

```bash
find "$tmpdir" -type f -name "*.wav" ! -name "._*" -print0 | xargs -0 -I{} mv -n "{}" "${H2T_DATADIR}/winoST/audio/en/"
rm -rf "$tmpdir" "${H2T_DATADIR}/winoST/WinoST.zip"
```

### Download WinoMT dataset

Download the aggregated English references from the **mt\_gender** repository:

```bash
wget https://github.com/gabrielStanovsky/mt_gender/raw/refs/heads/master/data/aggregates/en.txt -O "${H2T_DATADIR}/winoST/en.txt"

wget https://github.com/gabrielStanovsky/mt_gender/raw/refs/heads/master/data/aggregates/en_anti.txt -O "${H2T_DATADIR}/winoST/en_anti.txt"

wget https://github.com/gabrielStanovsky/mt_gender/raw/refs/heads/master/data/aggregates/en_pro.txt -O "${H2T_DATADIR}/winoST/en_pro.txt"
```

### Generate Input Jsons

Run the Python script to generate the processed data:

```bash
python generate.py
```

## Expected Output

After running the steps above, your directory layout will be:

```
${H2T_DATADIR}/
└─ winoST/
   ├─ audio/
   │  └─ en/
   │     ├─ 0001.wav
   │     ├─ 0002.wav
   │     └─ ... (total: 3,888 WAV files)
   ├─ en.txt
   ├─ en_anti.txt
   └─ en_pro.txt
```

If your generate.py script writes manifests, you should get JSONL files (one per language pair) under your chosen output path (e.g., ./manifests/winoST/). A jsonl entry looks like:

```json
{
  "dataset_id": "WinoST",
  "sample_id": 1,
  "src_audio": "/winoST/audio/en/0001.wav",
  "src_ref": "The developer argued with the designer because she did not like the design.",
  "tgt_ref": null,
  "src_lang": "en",
  "tgt_lang": "it",
  "benchmark_metadata": {
    "gender": "female",
    "profession": "developer",
    "label": "1",
    "stereotype": "anti",
    "context": "short"
  }
}
```

## License

WinoST is available under the [MIT License](https://github.com/gabrielStanovsky/mt_gender/blob/master/LICENSE) with the limitation that recordings cannot be used for speech synthesis, text to speech, voice conversion or other applications where the speaker voice is imitated or reproduced.