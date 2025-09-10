"""Prepare jsonl files for Mandi poem and short story"""
import argparse
import json
import os
import re
import soundfile as sf

src_lang = 'zh'
#As we do english centric testing, we translate in only zh-en direction
tgt_lang = 'en'

base_dir = os.environ.get("H2T_DATADIR")

VALID_TXT = ['nws','wch']

def get_zh(line):
    return ''.join(re.findall(r'[\u3400-\u4DBF\u4E00-\u9FFF]+', line))

def get_transcripts(in_dir):
    """
    nws = short story
    wch = poem
    """
    out = {}
    for txt_type in VALID_TXT:
        txt_file = os.path.join(in_dir, 'mfa_lexicon (reading materials)',f'mandarin_{txt_type}.txt')
        with open(txt_file,'r',encoding='utf-8') as f:
            temp = " ".join([get_zh(line).strip() for line in f])
            out[txt_type]=temp

    return out

def build_jsonl(in_dir, transcripts):
    audio_dir = os.path.join(in_dir,'mfa_wav')
    
    audio_output_dir = os.path.join(base_dir, 'mandi', 'audio', src_lang)
    os.makedirs(audio_output_dir,exist_ok=True)

    json_out = f"manifests/mandi/{src_lang}-{tgt_lang}.jsonl"

    with open(json_out,'w',encoding='utf-8') as f:
        for i, a in enumerate(os.listdir(audio_dir)):
            a_split = a.split('.')[0].split("_")
            data = {'native_acc':a_split[0],
                    'spoken_acc':a_split[3],
                    'gender':a_split[2],
                    'participant_id':a_split[1],
                    'txt':a_split[4].lower()}
            if data['txt'] not in VALID_TXT:
                continue

            utt_id = f'{i}'.zfill(3)

            audio_filename=f"{utt_id}.wav"
            audio_filepath = os.path.join(audio_output_dir, audio_filename)
            relative_audio_path = f"/mandi/audio/{src_lang}/{audio_filename}"
            
            #save record
            record = {
                "dataset_id": "mandi",
                "sample_id": utt_id,
                "src_audio": relative_audio_path,
                "src_ref": transcripts[data['txt']],
                "tgt_ref": None,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "benchmark_metadata": {
                    "native_acc": data.get('native_acc'),
                    "spoken_acc": data.get("spoken_acc"),
                    "participant_id": data.get("participant_id"),
                    "context": "short"
                }
            }

            #save audio
            if not os.path.exists(audio_filepath):
                orig_path = os.path.join(in_dir,'mfa_wav',a)
                audio, sr = sf.read(orig_path)

                sf.write(
                    audio_filepath,
                    audio,
                    sr
                )

            #Add json line
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir',type=str,required=True,
                            help='local path to the directory downloaded from here: https://osf.io/fgv4w/files/osfstorage (containing texts and audio)')

    args = parser.parse_args()

    transcripts = get_transcripts(args.in_dir)

    build_jsonl(args.in_dir, transcripts)