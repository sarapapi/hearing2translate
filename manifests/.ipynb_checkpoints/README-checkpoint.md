# Evaluation Manifests

This folder contains prepared **JSONL manifests** for evaluating SpeechLLMs.  
Each manifest follows a unified schema so that different datasets can be compared consistently.

---

## Schema

```json
{
  "dataset_id": "string",        // source dataset ID
  "sample_id": 0,                // unique sample index
  "src_audio": "path_or_url",    // audio file (if available)
  "src_ref_text": "string|null", // source transcription
  "tgt_ref": "string|dict|null", // reference translation(s), if available
  "src_lang": "iso_code",        // source language (ISO-639)
  "ref_lang": "iso_code|null",   // target language (ISO-639)
  "benchmark_metadata": { ... }  // dataset-specific metadata
   //phenomenon-specific
   "country": "string|null",      // speaker L1 / accent category
   "doc_id",                     // 
}