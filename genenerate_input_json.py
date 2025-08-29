import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


import jsonlines
from datasets import load_dataset
from dotenv import load_dotenv

dataset_types = ["standard","gender"]
class DatasetType(Enum):
    STANDARD = 1
    GENDER = 2
    CODESWITCH = 3
    CONVERSATION = 4
    LONGFORM = 5
    GENDERBIAS = 6
    TOXICITY = 7
    DIALECTACCENT = 8
    TERMINOLOGY = 9
    NONNATIVE = 10
    #DISFLUENCIES = 
    #NOISY = 
langs = [""]

@dataclass
class InputJson:
    dataset_id: str
    dataset_type: list
    sample_id: int
    src_audio: Path | None
    src_ref: str | None
    tgt_ref: str | None | dict
    src_lang: str
    ref_lang: str
    benchmark_metadata: dict | None


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

    @classmethod
    def generate(cls):
        dataset = load_dataset("byan/cs-fleurs")
        dataset_path = Path(f"./datasets/{cls.dataset_id}")
        dataset_path.mkdir(parents=True, exist_ok=True)
        lang_pair = set(dataset["test"]["language"])
        for lp in lang_pair:
            src, tgt = lp.split("-")
            samples = []
            #TODO Ask if json should be separated by json or if maybe we shoudl reconsider it
            dataset_path_json = dataset_path / f"{src}-{tgt}.jsonl"
            with jsonlines.open(dataset_path_json, mode="w") as writer:
                for i, sample in enumerate(dataset["test"].filter(lambda x : x["language"] == lang_pair)):
                    print(sample)
                    samples.append(
                            InputJson(
                                dataset_id=cls.dataset_id,
                                dataset_type= DatasetType.CODESWITCH,
                                sample_id=i, #TODO Over what for loop do we save the sample_id? Whole dataset o rlang_pair?
                                src_audio=f"/read/test/audio/{src}/{id}.wav",
                                src_ref="",
                                tgt_ref="",
                                src_lang= src,
                                ref_lang= tgt,
                                benchmark_metadata={"cs_lang" : [src,tgt]},
                            )
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
