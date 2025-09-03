import os
from dataclasses import asdict
from pathlib import Path

from json_schema import InputJson, DatasetType

import jsonlines
from tqdm import tqdm
from datasets import load_dataset, Dataset
from langcodes import Language
from dotenv import load_dotenv
import pandas as pd 

load_dotenv()
langs = {"cmn-eng", "deu-eng", "fra-eng", "spa-eng"}

def generate_csfleurs():
    print("Generating cs_fleurs dataset")
    dataset_id = "cs_fleurs"

    dataset = load_dataset("byan/cs-fleurs", split="test")
    dataset_path = Path(f"{os.environ['H2T_DATADIR']}/{dataset_id}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    lang_pair = set(dataset["language"]) & langs
    for lp in tqdm(lang_pair):
        src, tgt = lp.split("-")
        #Normalize Mandarin Chinese to be consistent with other datasets
        src_iso = src
        if src == "cmn":
            src = "zh"
        src, tgt = Language.get(src).language, Language.get(tgt).language
        dataset_path_json = dataset_path / f"{src}-{tgt}.jsonl"
        #Awful casting into pandas because Huggignface dataset does not like joining of datasets
        dataset_fleurs = Dataset.from_pandas(pd.DataFrame(load_dataset("google/fleurs", f'en_us', split="test")).drop_duplicates("id").sort_values(by="id"))
        min_id = dataset_fleurs[0]["id"]
        with jsonlines.open(dataset_path_json, mode="w") as writer:
            samples = []
            for i, sample in enumerate(tqdm(dataset.filter(lambda x : x["language"] == lp))):
                ids = int(sample["id"].split("_")[1]) 
                samples.append(
                        asdict(InputJson(
                            dataset_id=dataset_id,
                            sample_id=i,
                            src_audio=f"{dataset_id}/data/read/test/audio/{src_iso}/{sample['id']}.wav",
                            src_ref=sample["text"],
                            tgt_ref=dataset_fleurs[ids - min_id]["raw_transcription"], #1600
                            src_lang= src,
                            ref_lang= tgt,
                            benchmark_metadata={"cs_lang" : [src,tgt], "context" : "short", "dataset_type" : DatasetType.CODESWITCH }
                        ))
                        )
            writer.write_all(samples)

if __name__ == "__main__":
    generate_csfleurs()