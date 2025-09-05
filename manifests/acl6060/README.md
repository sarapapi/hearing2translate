# ACL-6060

## Overview
This tool generates both short- and long-form manifests for the **ACL-6060** dataset by combining:

- **Hugging Face dataset**: [`ymoslem/acl-6060`](https://huggingface.co/datasets/ymoslem/acl-6060) (split: `eval`)  
  Provides segmented audio, transcripts, and translations.  
- **Manually downloaded files**: [GitHub release](https://github.com/sarapapi/hearing2translate/releases/tag/data-share-acl6060)  
  `manifests/acl6060/long/<lang>.xml` defines document boundaries and long-form transcripts.

---

## Pipeline
1. **Build mapping**  
   Parse `long/en.xml` to build a `seg_id → doc_id` mapping.  
   - Note: `seg_id` in XML corresponds to `sample_id - 1` in the HF dataset.

2. **Short-form records**  
   - Iterate through the HF dataset.  
   - Save each utterance as:  
     ```
     {H2T_DATADIR}/acl6060/audio/en/{sample_id}.wav
     ```  
   - Write one JSONL line per target language (`context = "short"`) to:  
     ```
     manifests/acl6060/en-<lang>.jsonl
     ```

3. **Long-form records**  
   - Assign one global long-form `sample_id` per `doc_id` (shared across all languages).  
   - For each language:  
     - Parse `<lang>.xml`.  
     - Concatenate segment texts per `docid` with `"\n"`.  
     - Append one JSONL line (`context = "long"`) to `en-<lang>.jsonl`.

4. **Mapping file**  
   Create a mapping file linking each `doc_id` to its assigned long-form `sample_id` and the corresponding long audio filename.  

   > **Note:** Long-form audio files must be manually downloaded from the [GitHub release](https://github.com/sarapapi/hearing2translate/releases/tag/data-share-acl6060).  
   > Place them under:  
   > ```
   > H2T_DATADIR/acl6060/audio/en/
   > ```  
   > Then rename the files according to the mapping provided in:  
   > ```
   > manifests/acl6060/long_audio_mapping.txt
   > ```

---

## Outputs
- **Short-form manifests**  
  - `manifests/acl6060/en/en-de.jsonl`  
  - `manifests/acl6060/en/en-fr.jsonl`  
  - `manifests/acl6060/en/en-pt.jsonl`  
  - `manifests/acl6060/en/en-zh.jsonl`

- **Long-form manifests**  
  - Appended at the end of each JSONL file.

- **Audio mapping file**  
  - `manifests/acl6060/long_audio_mapping.txt`

---

## Environment
- Required environment variable: `H2T_DATADIR`  
- Install dependencies:  
  ```bash
  pip install "datasets[audio]<4.0" soundfile