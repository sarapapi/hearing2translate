import os
import json
import soundfile as sf
from datasets import load_dataset

# --- Configuration ---
# Define the language pairs you want to process.
# The format is 'source_language-target_language'.
# These should be two-letter ISO 639-1 language codes.
LANGUAGE_PAIRS = [
    "en-de",
    "en-fr",
    "en-pt",
    "en-nl",
    "en-it",
    "en-es",
    "en-zh",
]




# Define the splits to be used from the dataset
# The Libristutter dataset on Hugging Face uses 'train' and 'test'.
SPLITS = ["test"]


def process_libristutter_dataset():
    """
    Downloads and processes the Libristutter dataset.
    For each split (train, test), it creates a .jsonl file
    with metadata and saves the source audio files to a local directory. 
    """

    print("Starting Libristutter dataset processing...")

    try:
        # Create output directory for audio files (once per source language)
        audio_output_dir = os.path.join(os.environ['H2T_DATADIR'], "libristutter", "audio","en")
        os.makedirs(audio_output_dir, exist_ok=True)
        print(f"Audio files  will be saved in: '{audio_output_dir}'")

        # Loop over each split to create separate files
        for split in SPLITS:
            print(f"--- Processing split: {split} ---")
            print(f"Loading dataset for split: {split}")
            dataset = load_dataset("stillerman/libristutter-4.7k", split=split, trust_remote_code=True)
            print(f"Loaded {len(dataset)} samples in split '{split}'.")
            
            records  = []
            for sample in dataset:
                sample_id = sample['name']
 
                # Define the audio file path
                audio_filename = sample['audio']['path'].split('/')[-1].replace('.flac','.wav') #f"{sample_id}.wav"
                audio_filepath = os.path.join(audio_output_dir, audio_filename)
                relative_audio_path = os.path.join("libristutter", "audio","en", audio_filename)

                # Save the audio file only if it doesn't already exist
                if not os.path.exists(audio_filepath):
                    
                    sf.write(
                        audio_filepath,
                        sample["audio"]["array"],
                        sample["audio"]["sampling_rate"]
                    )

                # Save stutter position and remove [STUTTER] tag from text
                stutter_pos = [i for i, x in enumerate(sample["text"].split()) if x == "[STUTTER]"]
                text = sample["text"].replace("[STUTTER]","")

                # Construct the JSON record
                record = {
                    "dataset_id": "libristutter",
                    "sample_id": sample_id,
                    "src_audio": f"/{relative_audio_path}",
                    "src_ref": text,
                    "tgt_ref": "null",
                    "src_lang": "en",
                    "tgt_lang": "en",
                    "benchmark_metadata": {
                        "has_stutter": "True" if not len(stutter_pos) == 0 else "False",
                        "stutter_pos": stutter_pos,
                        "context": "short"
                    }
                }
                records.append(record)

            #Write a jsonl file for each translation direction
            for lang_pair in LANGUAGE_PAIRS:
                jsonl_filename = f"./manifests/libristutter/{lang_pair}.jsonl"
                tgt_lang = lang_pair.split('-')[1]
                with open(jsonl_filename, 'w', encoding='utf-8') as f:
                    #records = map(lambda r:r['tgt_lang'] = tgt_lang , records)
                    records_written = 0
                    # Write the JSONL records to the file 
                    for record in records:
                        record['tgt_lang'] = tgt_lang
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        records_written += 1
                    print(f"Successfully created '{jsonl_filename}' with {records_written} records.")


    except Exception as e:
        print(f"An error occurred while processing dataset': {e}")

    print("\nDataset processing finished.")

if __name__ == "__main__":
    
    process_libristutter_dataset()
