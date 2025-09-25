import json
import os

def open_file(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f.readlines()]

def safe_doc_id(long_path: str) -> str:
    base = os.path.basename(long_path)
    return os.path.splitext(base)[0]

def main():
    root = "../mcif-short/tmp"
    tgt_langs = ["de", "it", "zh"]
    id_mapping = "id_mapping.jsonl"

    for tgt_lang in tgt_langs:
        mappings = open_file(id_mapping) # [{iid, long_path, short_path: [...]}}
        item_group = open_file(os.path.join(root, f"grouped-{tgt_lang}.jsonl")) # [{'audio_path': [...], 'transcript': [...], 'reference': [...]}] Hid by .gitignore

        # short_path -> item idx
        short_to_item = {}
        for it in item_group:
            for p in it["audio_path"]:
                short_to_item[p] = it

        new_audio_mapping = {}
        manifests = []
        audio_id = 0

        for m in mappings:
            short_path = m["short_path"][0]
            doc_id = safe_doc_id(m["long_path"])
            item = short_to_item[short_path]

            for seg_id, (src, ref) in enumerate(zip(item["transcript"], item["reference"])):
                manifests.append({
                    "doc_id": doc_id,
                    "seg_id": seg_id,
                    "src_ref": src.strip(),
                    "tgt_ref": ref.strip(),
                })

            for old_audio in item["audio_path"]:
                new_audio_mapping[old_audio] = {"audio_path": f"{audio_id}.wav", "doc_id": doc_id}
                audio_id += 1

        with open(f"segmented_en-{tgt_lang}.jsonl", "w", encoding="utf-8") as f:
            for line in manifests:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                
    # English audios
    with open(f"audio_path.json", "w", encoding="utf-8") as f:
        json.dump(new_audio_mapping, f, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    main()