"""A script to generate the CommonAccent jsonl files"""
import json
import os

from datasets import concatenate_datasets, load_dataset
import pandas as pd
import soundfile as sf

#As we have no target references anyways, we translate to all of them
tgt_langs = [
    'es','nl'
    'fr','de',
    'zh','it',
    'pt'
]

#Src langs in comAcc
src_langs = [
    'it','en',
    'de','es'
]

def build_jsonl():
    """Build the jsonl files containing datapoints for each src-tgt pair in CommonAccent"""
    for src in src_langs:
        #Get comAcc specific data points
        df = pd.read_csv(f'./eval_csv/{src}_test.csv', encoding='utf-8', header=0)
        df['utt_id'] = df['utt_id'].astype(str)
        #load common voice:
        cv = load_dataset("mozilla-foundation/common_voice_11_0", src, streaming=False, split=['train','validation','test'])
        cv = concatenate_datasets(cv)

        cv = cv.add_column('utt_id', [path.split("_")[-1].split('.')[0] for path in cv['path']])

        valid_utt_ids = set(df['utt_id'].values)
        # Filter the dataset
        cv_filtered = cv.filter(
            lambda batch: [utt_id in valid_utt_ids for utt_id in batch['utt_id']], 
            batched=True, 
            batch_size=10000 
        )
        audio_output_dir = os.path.join("audio", src)
        os.makedirs(audio_output_dir, exist_ok=True)

        #English centric lang pairs
        if src == 'en':
            for tgt in tgt_langs:
                json_out = f"{src}-{tgt}.jsonl"
                with open(json_out,'w',encoding='utf-8') as f:
                    for row in cv_filtered:
                        #If the path to the audio doesn't exist, save it
                        audio_filename=f"{row['utt_id']}.wav"
                        audio_filepath = os.path.join(audio_output_dir, audio_filename)
                        relative_audio_path = f"./{audio_filepath.replace(os.sep, '/')}"

                        if not os.path.exists(audio_filepath):
                            sf.write(
                                audio_filepath,
                                row["audio"]["array"],
                                row["audio"]["sampling_rate"]
                            )

                        record = {
                            "dataset_id": "commonAccent",
                            "sample_id": row['utt_id'],
                            "src_audio": relative_audio_path,
                            "src_ref": row["sentence"],
                            "tgt_ref": None,
                            "src_lang": src,
                            "tgt_lang": tgt,
                            "benchmark_metadata": {
                                "acc": df[df['utt_id'] == row['utt_id']]['accent'].iloc[0]
                            }
                        }

                        #Add json line
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        else:
            #With non-english src languages, we translate into en only
            tgt = 'en'
            json_out = f"{src}-{tgt}.jsonl"
            with open(json_out,'w',encoding='utf-8') as f:
                for row in cv_filtered:
                    #If the path to the audio doesn't exist, save it
                    audio_filename=f"{row['utt_id']}.wav"
                    audio_filepath = os.path.join(audio_output_dir, audio_filename)
                    relative_audio_path = f"./{audio_filepath.replace(os.sep, '/')}"

                    if not os.path.exists(audio_filepath):
                        sf.write(
                            audio_filepath,
                            row["audio"]["array"],
                            row["audio"]["sampling_rate"]
                        )

                    record = {
                        "dataset_id": "commonAccent",
                        "sample_id": row['utt_id'],
                        "src_audio": relative_audio_path,
                        "src_ref": row["sentence"],
                        "tgt_ref": None,
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "benchmark_metadata": {
                            "acc": df[df['utt_id'] == row['utt_id']]['accent'].iloc[0]
                        }
                    }

                    #Add json line
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')            


if __name__=="__main__":
    build_jsonl()