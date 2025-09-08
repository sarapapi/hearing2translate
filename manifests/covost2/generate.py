"""A script to generate the CoVoST2 jsonl files and save audios"""
import argparse
import csv
import json
import os
import time

from datasets import concatenate_datasets, load_dataset
import pandas as pd
import soundfile as sf

# As we have no target references anyways, we translate to all of them
tgt_langs_covost = [
    'de', 'zh'
]

tgt_langs_noRef = ['es','nl','fr','it','pt']

src_langs = [
    'es', 'de', 'en',
    'pt','it', 'zh'
]

base_dir = os.environ.get("H2T_DATADIR")

def load_written_ids(json_out):
    """Read existing JSONL file and collect sample_ids to avoid duplicates"""
    if not os.path.exists(json_out):
        return set()
    written = set()
    with open(json_out, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                written.add(rec["sample_id"])
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return written

def process_in_batches(dataset, batch_size=1000):
    """Process dataset in batches"""
    batch = []
    for _, row in enumerate(dataset):
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
            # Small delay to prevent overwhelming the connection
            time.sleep(0.1)
    # Process remaining items
    if batch:
        yield batch


def build_jsonl(clip_dir):
    """Build the jsonl files containing datapoints for each src-tgt pair in CoVoST2"""

    for src in src_langs:
        audio_output_dir = os.path.join(base_dir, 'covost2', "audio", src)
        os.makedirs(audio_output_dir, exist_ok=True)

        if src == 'en':
            for tgt in tgt_langs_covost:
                json_out = f"manifests/covost2/{src}-{tgt}.jsonl"
                process_language_pair(src, tgt, json_out, audio_output_dir, clip_dir, ref = True)
            for tgt in tgt_langs_noRef:
                # We run inference anyways even without target reference
                json_out = f"manifests/covost2/{src}-{tgt}.jsonl"
                process_language_pair(src, tgt, json_out, audio_output_dir, clip_dir, ref = False)
        else:
            tgt = 'en'
            json_out = f"manifests/covost2/{src}-{tgt}.jsonl"
            process_language_pair(src, tgt, json_out, audio_output_dir, clip_dir)


def process_language_pair(src, tgt, json_out, audio_output_dir, clip_dir, ref = True):
    """Process a single language pair with improved error handling"""
    
    # Get covost test points
    if ref==True:
        df = pd.read_csv(f'./manifests/covost2/translations/covost_v2.{src}_{tgt}.tsv', 
                        encoding='utf-8', header=0, sep='\t',
                        escapechar='\\', quoting=csv.QUOTE_NONE, na_filter=False)
    else:
        #just always use en-de since the source audio will be the same
        df = pd.read_csv(f'./manifests/covost2/translations/covost_v2.en_de.tsv', 
                encoding='utf-8', header=0, sep='\t',
                escapechar='\\', quoting=csv.QUOTE_NONE, na_filter=False)
    df = df[df['split'] == 'test']
    print(f"Now processing {src}-{tgt}: {len(df)} samples")
    
    df['utt_id'] = df['path'].apply(lambda x: x.replace('.mp3', '').split('_')[-1])
    df['utt_id'] = df['utt_id'].astype(str)
    
    # Create a lookup dictionary for faster access
    translation_lookup = dict(zip(df['utt_id'], df['translation']))
    valid_utt_ids = set(df['utt_id'].values)
    
    # Load already written IDs
    written_ids = load_written_ids(json_out)
    print(f"Found {len(written_ids)} already processed samples")
    
    if src in ['zh','pt','it']:
        # Load common voice dataset
        if src == 'zh':
            src_tag = 'zh-CN'
        else:
            src_tag = src
        cv = load_dataset("mozilla-foundation/common_voice_4_0", src_tag, streaming=True, split='test')
    
        processed_in_session = 0
        
        with open(json_out, 'a', encoding='utf-8') as f:
            for batch in process_in_batches(cv):
                for row in batch:
                    utt_id = row['audio']['path'].split('_')[-1].split('.')[0]
                    
                    # Filter the dataset
                    if utt_id not in valid_utt_ids:
                        continue
                    if utt_id in written_ids:
                        continue
                    
                    # Get translation using lookup
                    if utt_id not in translation_lookup:
                        print(f"Warning: No translation found for {utt_id}")
                        continue
                    
                    # Save audio file
                    audio_filename = f"{utt_id}.wav"
                    audio_filepath = os.path.join(audio_output_dir, audio_filename)
                    relative_audio_path = f"/covost2/audio/{src}/{audio_filename}"
                    
                    if not os.path.exists(audio_filepath):
                        sf.write(
                            audio_filepath,
                            row["audio"]["array"],
                            row["audio"]["sampling_rate"]
                        )
                    
                    record = {
                        "dataset_id": "covost2",
                        "sample_id": utt_id,
                        "src_audio": relative_audio_path,
                        "src_ref": row["sentence"],
                        "tgt_ref": translation_lookup[utt_id],
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "benchmark_metadata": {
                            "context": 'short'
                        }
                    }
                    
                    # Write record
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written to disk
                    
                    # Update written_ids to prevent duplicates in current session
                    written_ids.add(utt_id)
                    processed_in_session += 1
                    
                    if processed_in_session % 100 == 0:
                        print(f"Processed {processed_in_session} samples for {src}-{tgt}")
    else:
        #Use downloaded CV4 from in dir
        processed_in_session=0
        metadata = pd.concat([pd.read_csv(f'{clip_dir}/{src}/{split}.tsv', 
                     encoding='utf-8', header=0, sep='\t',
                     escapechar='\\', quoting=csv.QUOTE_NONE, na_filter=False) for split in ['train','dev','test','invalidated','other','validated']])
        metadata['utt_id'] = metadata['path'].apply(lambda x: x.replace('.mp3', '').split('_')[-1])
        metadata['utt_id'] = metadata['utt_id'].astype(str)
        with open(json_out, 'a', encoding='utf-8') as f:
            for i in range(len(df)):
                row = df.iloc[i]
                utt_id = str(row['utt_id'])
                if utt_id in written_ids:
                    continue

                audio, sr = sf.read(f'{clip_dir}/{src}/clips/common_voice_{src}_{utt_id}.mp3')
                # Save audio file
                audio_filename = f"{utt_id}.wav"
                audio_filepath = os.path.join(audio_output_dir, audio_filename)
                relative_audio_path = f"/covost2/audio/{src}/{audio_filename}"

                if not ref and src == 'en':
                    assert tgt in tgt_langs_noRef
                    record = {
                            "dataset_id": "covost2",
                            "sample_id": utt_id,
                            "src_audio": relative_audio_path,
                            "src_ref": metadata[metadata['utt_id'] == utt_id]['sentence'].iloc[0],
                            "tgt_ref": None,
                            "src_lang": src,
                            "tgt_lang": tgt,
                            "benchmark_metadata": {
                                "context": 'short'
                            }
                    } 
                else:
                    record = {
                            "dataset_id": "covost2",
                            "sample_id": utt_id,
                            "src_audio": relative_audio_path,
                            "src_ref": metadata[metadata['utt_id'] == utt_id]['sentence'].iloc[0],
                            "tgt_ref": translation_lookup[utt_id],
                            "src_lang": src,
                            "tgt_lang": tgt,
                            "benchmark_metadata": {
                                "context": 'short'
                            }
                    } 
                if not os.path.exists(audio_filepath):
                    sf.write(
                        audio_filepath,
                        audio,
                        sr
                    )
                # Write record
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written to disk
                
                # Update written_ids to prevent duplicates in current session
                written_ids.add(utt_id)
                processed_in_session += 1
                
                if processed_in_session % 100 == 0:
                    print(f"Processed {processed_in_session} samples for {src}-{tgt}")  
        
    print(f"Successfully completed {src}-{tgt} with {processed_in_session} new samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir",type=str,
                        help="dir containing the de, en, and es CommonVoice 4.0 data. Should contain 'en/clips','es/clips','de/clips'")
    args = parser.parse_args()
    build_jsonl(args.clip_dir)