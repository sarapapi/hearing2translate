# Evaluation Manifests

## Schema
Each line in a manifest follows this schema:

```jsonc
{
  "dataset_id": "string",        // source dataset ID
  "sample_id": 0,                // unique sample index
  "src_audio": "path_or_url",    // audio file (if available)
  "src_ref": "string|null",      // source transcription
  "tgt_ref": "string|dict|null", // reference translation(s), if available
  "src_lang": "iso_code",        // source language (ISO-639)
  "ref_lang": "iso_code|null",   // target language (ISO-639)
  "benchmark_metadata": { ... }  // dataset-specific metadata
}
```

## Available Manifests

| File                        | Dataset                   | Samples | Subset     | Src Lang | Ref Lang(s)        | References | Notes |
|-----------------------------|---------------------------|---------|------------|----------|--------------------|------------|-------|
| `./non_native/elitr_en-de.jsonl` | IWSLT 2020 Non-Native SLT | 1,917   | Dev+Test   | en       | de                 | All     | Non-native English → German |
| `./longform/acl6060_en-many.jsonl`     | ACL 60/60 (Long-form)     | 884     | All        | en       | de, fr, pt, zh     | All (multi-ref dict) | 32 unique docs; multi-target translations |
| `./accents/edacc_en.jsonl`           | EdAcc (Accented English)  | 19,144  | Dev+Test   | en       | —                  | —          | Accent & speaker metadata (country, gender) |
