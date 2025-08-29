import os
from abc import ABC
from dataclasses import dataclass, asdict
from pathlib import Path

from json_schema import InputJson, OutputJson, DatasetType

from langcodes import Language
import jsonlines
from datasets import load_dataset
from dotenv import load_dotenv

langs = ["es", "fr", "de", "zh",]

@dataclass
class Dataset(ABC):
    dataset_id: str

    @classmethod
    def generate(cls):
        pass

    @classmethod
    def load(cls):
        pass


class CSFleurs(Dataset):
    dataset_id = "cs_fleurs"

    #TODO: CSFLEURS is missing the corresponding FLEURS reference.
    @classmethod
    def generate(cls):
        dataset = load_dataset("byan/cs-fleurs", split="test")
        dataset_path = Path(f"./manifests/{cls.dataset_id}")
        dataset_path.mkdir(parents=True, exist_ok=True)
        lang_pair = set(dataset["language"])
        for lp in lang_pair:
            src, tgt = lp.split("-")
            dataset_path_json = dataset_path / f"{src}-{tgt}.jsonl"
            #dataset_fleurs = load_dataset("google/fleurs", f"{Language.get(src)}_{Language.get(tgt)}", split="train", streaming=True)
            with jsonlines.open(dataset_path_json, mode="w") as writer:
                samples = []
                for i, sample in enumerate(dataset.filter(lambda x : x["language"] == lp)):
                    samples.append(
                            asdict(InputJson(
                                dataset_id=cls.dataset_id,
                                sample_id=i,
                                src_audio=f"/read/test/audio/{src}/{sample['id']}.wav",
                                src_ref=sample["text"],
                                tgt_ref="",
                                src_lang= src,
                                ref_lang= tgt,
                                benchmark_metadata={"cs_lang" : [src,tgt], "dataset_type" : DatasetType.CODESWITCH }
                            ))
                            )
                writer.write_all(samples)


class EuroparlST(Dataset):
    benchmark_name = "europarl_st"

    @classmethod
    def generate(cls):
        pass


def generate_europarl_st():
    path = os.environ.get("europarl_st_path")
    langs = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"]
    paths = Path()


if __name__ == "__main__":
    CSFleurs.generate()
