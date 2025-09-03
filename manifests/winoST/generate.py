import os
import json

# --- Configuration ---
# Define the language pairs you want to process.
# The format is 'source_language-target_language'.
# These should be two-letter ISO 639-1 language codes.
LANGUAGE_PAIRS = [
    "en-fr",
    "en-it",
    "en-es",
]

def process_winost_dataset():

    audio_output_dir = './extracted/WinoST'
    txt_file_path = './en.txt'

    sentences_anti = [line.split("\t")[2] for line in open("./en_anti.txt", encoding="utf-8")]
    sentences_pro  = [line.split("\t")[2] for line in open("./en_pro.txt", encoding="utf-8")]
    
    for pair in LANGUAGE_PAIRS:
        src_lang, tgt_lang = pair.split('-')
        jsonl_filename = f"{src_lang}-{tgt_lang}.jsonl"

        with open(jsonl_filename, 'w', encoding='utf-8') as f_json:

            with open(txt_file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):

                    audio_filename = f"{idx+1:04d}.wav"
                    audio_filepath = os.path.join(audio_output_dir, audio_filename)

                    parts = line.strip().split("\t")
                    gender, label, sentence, target = parts

                    stereotype = None
                    if sentence in sentences_anti:
                        stereotype= 'anti'
                    elif sentence in sentences_pro:
                        stereotype= 'pro'

                    # Construct the JSON record
                    record = {
                        "dataset_id": "WinoST",
                        "sample_id": idx+1,
                        "src_audio": audio_filepath,
                        "src_ref": sentence,
                        "tgt_ref": None,
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "benchmark_metadata": {
                            "gender": gender, "profession": target, "label": label, "stereotype": stereotype, "context": "short"
                        }
                    }

                    f_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            

if __name__ == "__main__":
    
    process_winost_dataset()