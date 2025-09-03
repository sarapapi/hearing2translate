"""A script to generate the CoVoST2 jsonl files and save audios"""
import csv
import json
import os

from datasets import concatenate_datasets, load_dataset
import pandas as pd
import soundfile as sf

#As we have no target references anyways, we translate to all of them
tgt_langs = [
    'de','zh'
]

#Src langs in comAcc
src_langs = [
    'it','en',
    'de','es',
    'zh','pt'
]

def build_jsonl():
    """Build the jsonl files containing datapoints for each src-tgt pair in CoVoST2"""

    for src in src_langs:

        audio_output_dir = os.path.join("audio", src)
        os.makedirs(audio_output_dir, exist_ok=True)

        if src == 'en':
            for tgt in tgt_langs:
                json_out = f"{src}-{tgt}.jsonl"
                #Get covost test points:
                df = pd.read_csv(f'./translations/covost_v2.{src}_{tgt}.tsv', encoding='utf-8', header=0, sep='\t',
                                    escapechar='\\',quoting=csv.QUOTE_NONE, na_filter=False)
                df = df[df['split']=='test']
                print(len(df))
                df['utt_id'] = df['path'].apply(lambda x: x.replace('.mp3','').split('_')[-1])
                df['utt_id'] = df['utt_id'].astype(str)

                # #load common voice:
                cv = load_dataset("mozilla-foundation/common_voice_4_0", src, streaming=True, split='test')
                valid_utt_ids = set(df['utt_id'].values)
                with open(json_out,'w',encoding='utf-8') as f:
                    for row in cv:
                        utt_id = row['audio']['path'].split('_')[-1].split('.')[0]
                        # print(row)
                        # print(utt_id)
                        
                        # Filter the dataset
                        if utt_id in valid_utt_ids:
                            #If the path to the audio doesn't exist, save it
                            audio_filename=f"{utt_id}.wav"
                            audio_filepath = os.path.join(audio_output_dir, audio_filename)
                            relative_audio_path = f"/covost2/{audio_filepath.replace(os.sep, '/')}"

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
                                "tgt_ref": df[df['utt_id'] == utt_id]['translation'].iloc[0],
                                "src_lang": src,
                                "tgt_lang": tgt,
                                "benchmark_metadata": {
                                    "context": 'short'
                                }
                            }

                            #Add json line
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        else:
            tgt = 'en'
            json_out = f"{src}-{tgt}.jsonl"
            #Get covost test points:
            df = pd.read_csv(f'./translations/covost_v2.{src}_{tgt}.tsv', encoding='utf-8', header=0, sep='\t',
                                escapechar='\\',quoting=csv.QUOTE_NONE, na_filter=False)
            df = df[df['split']=='test']
            print(len(df))
            df['utt_id'] = df['path'].apply(lambda x: x.replace('.mp3','').split('_')[-1])
            df['utt_id'] = df['utt_id'].astype(str)

            # #load common voice:
            cv = load_dataset("mozilla-foundation/common_voice_4_0", src, streaming=True, split='test')
            valid_utt_ids = set(df['utt_id'].values)
            with open(json_out,'w',encoding='utf-8') as f:
                for row in cv:
                    utt_id = row['audio']['path'].split('_')[-1].split('.')[0]
                    # print(row)
                    # print(utt_id)
                    
                    # Filter the dataset
                    if utt_id in valid_utt_ids:
                        #If the path to the audio doesn't exist, save it
                        audio_filename=f"{utt_id}.wav"
                        audio_filepath = os.path.join(audio_output_dir, audio_filename)
                        relative_audio_path = f"/covost2/{audio_filepath.replace(os.sep, '/')}"

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
                            "tgt_ref": df[df['utt_id'] == utt_id]['translation'].iloc[0],
                            "src_lang": src,
                            "tgt_lang": tgt,
                            "benchmark_metadata": {
                                "context": 'short'
                            }
                        }

                        #Add json line
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

        break


if __name__=="__main__":
    build_jsonl()