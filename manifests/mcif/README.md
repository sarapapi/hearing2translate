# MCIF Data Mapping

⚠️ Note: Do not use the files in this directory for model inference.
For model evaluation, use `segmented_{src_lang}-{tgt_lang}.jsonl`.

## Segmented Files
Each entry has the following structure:
```
{
  "doc_id": "<long-form audio path>",
  "seg_id": "<segment index within the document>",
  "src_ref": "<segmented transcript>",
  "tgt_ref": "<segmented translation>"
}
```

## Additional Files

- `audio_path.json` : a JSON file that maps the original audio paths to the new audio paths ({sample_id}.wav).
    ```
        {
          "<original_audio_path>": {
                "audio_path": "<sample_id>.wav",
                "doc_id": "<long-form audio path>"
          }
        }
    ```
    
- `id_mapping.jsonl` : Contains mapping information between long-form and short-form audio paths.
    ```
        {
          "iid": "<unique document ID>",
          "long_path": "<long-form audio path>.wav",
          "short_path": ["<short-form audio path 1>", "<short-form audio path 2>", ...]
        }
    ```
    
- `xml/` : Contains the original long-form and short-form XML files.
