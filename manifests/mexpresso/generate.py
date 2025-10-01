
import os
from dataclasses import asdict
import shutil
from pathlib import Path

#TODO: J.Iranzo. Quick hack to have this script work from both root folder amd inside the cs_fleurs folder
# since we need to import data_schema and currently we have a flat style python library, which
# does not like doing submodules. This should probably be changed!!!
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_schema import InputJson, DatasetType

import jsonlines
from tqdm import tqdm
from datasets import load_dataset, Dataset
from langcodes import Language
from dotenv import load_dotenv
import pandas as pd 
import subprocess

load_dotenv()
langs = {"cmn-eng", "deu-eng", "fra-eng", "spa-eng"}


def generate_mexpresso():
    dataset_id = "mexpresso"
    print(f"Generating {dataset_id} dataset")

    split = "test"
    dataset = load_dataset("jorirsan/mExpresso", split=split)
    dataset_path = Path(__file__).parent 
    (dataset_path / "audio").mkdir(parents=True, exist_ok=True)

    #TODO: Add the paths from huggingface cache. Should change it so that it is multiplatform
    cmd = r"huggingface-cli scan-cache | grep jorirsan/mExpresso | awk -F ' ' '{print $NF}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.rstrip()
    hf_data_folder=subprocess.run(f"echo {result}/snapshots/$(cat {result}/refs/main)", shell=True, capture_output=True, text=True).stdout.rstrip()
    print(hf_data_folder)

    langs = {"cmn","deu","fra","ita","spa"}
    #langs = set()
    non_tgt_langs = {"nl","pt"}

    for l in tqdm(langs | non_tgt_langs):
        if l in langs:
            dataset_subset = dataset.filter(lambda x : x["tgt_text"][l]).map(lambda x : x | {"tgt_text": x["tgt_text"][l] })
        else:
            dataset_subset = dataset

        if l == "cmn":
            l = "zh"
        src, tgt = Language.get("en").language, Language.get(l).language

        (dataset_path / "audio"/ src).mkdir(parents=True, exist_ok=True)
        dataset_path_json = dataset_path / f"{src}-{tgt}.jsonl"

        with jsonlines.open(dataset_path_json, mode="w") as writer:
            samples = []
            for i, sample in enumerate(tqdm(dataset_subset)):
                sample_path = (Path(__file__).parent / "audio" / src / sample['id']).with_suffix(".wav")
                sample_path_json = (Path(dataset_id) / "audio" / src / sample['id']).with_suffix(".wav")
                samples.append(
                        asdict(InputJson(
                            dataset_id=dataset_id,
                            sample_id=i,
                            src_audio=str(sample_path_json),
                            src_ref=sample["src_text"],
                            tgt_ref=sample["tgt_text"] if tgt not in non_tgt_langs else None, 
                            src_lang= src,
                            tgt_lang= tgt,
                            benchmark_metadata={"context" : "short", "emotion" : sample["label"], "dataset_type" : DatasetType.EMOTION }
                        ))
                        )
                if not sample_path.is_file():
                    sample_path.symlink_to(f"{hf_data_folder}/{split}/{sample['id']}.wav")
            writer.write_all(samples)
if __name__ == "__main__":
    generate_mexpresso()