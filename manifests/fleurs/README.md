# FLEURS

This script downloads selected splits of the **FLEURS** dataset from Hugging Face and builds, for each language pair and split, a JSONL file with source audio + aligned source/target transcriptions. It also exports the source-waveforms (`.wav`) to a local `audio/<src_lang>/` folder.

---

## What the script does

* Iterates over `LANGUAGE_PAIRS` like `en-de`, `fr-en`, etc.
* For each split in `SPLITS` (default: `test`):

  * Loads the corresponding FLEURS configs for source and target (e.g., `en_us`, `de_de`).
  * Writes `<src>-<tgt>.<split>.jsonl` containing one JSON record per sample with:

    * path to the exported **source audio** (`.wav`)
    * **source transcription** (FLEURS field `raw_transcription`)
    * **target transcription** looked up by the shared `id`
    * `src_lang`, `tgt_lang`, `benchmark_metadata` (currently `gender`)
  * Saves the **source audio** to `./audio/<src_lang>/<sample_id>.wav` if not already present.

### JSONL schema (per line)

```json
{
  "dataset_id": "fleurs",
  "sample_id": "<string>",
  "src_audio": "/fleurs/audio/<src_lang>/<sample_id>.wav",
  "src_ref": "<source raw_transcription>",
  "tgt_ref": "<target raw_transcription>",
  "src_lang": "<two-letter ISO 639-1>",
  "tgt_lang": "<two-letter ISO 639-1>",
  "benchmark_metadata": {"gender": "0|1"}
}
```

### Example output

```
.
├── audio/
│   ├── en/
│   │   ├── 1271.wav
│   │   ├── 1272.wav
│   │   └── ...
│   └── fr/
│       └── ...
├── en-de.jsonl
├── en-fr.jsonl
└── ...
```


## How to run

```bash
export H2T_DATADIR='' #Path where the Fleurs dataset is stored.

python generate.py
```

## What it generates

For each pair `<src>-<tgt>` you get:

* `./<src>-<tgt>.jsonl` – metadata + paths to audio.
* `/fleurs/audio/<src>/<sample_id>.wav` – exported source audio for that pair.