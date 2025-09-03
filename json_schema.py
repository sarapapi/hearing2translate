from dataclasses import dataclass, asdict
from enum import Enum, auto

@dataclass
class InputJson:
    dataset_id: str
    #dataset_type: list
    sample_id: int
    src_audio: str | None
    src_ref: str | None
    tgt_ref: str | None | dict
    src_lang: str
    ref_lang: str
    benchmark_metadata: dict | None #{"context":  ("short","long")}}


@dataclass
class OutputJson:
    dataset_id: str
    sample_id: str
    src_lang : str
    tgt_lang : str
    model : str
    output : str



class DatasetType(str, Enum):
    STANDARD = "standard"
    GENDER = "gender"
    CODESWITCH = "code_switch"
    CONVERSATION = "conversation"
    LONGFORM = "longform"
    GENDERBIAS = "gender_bias"
    TOXICITY = "toxicity"
    DIALECTACCENT = "accents"
    TERMINOLOGY = "terminology"
    NONNATIVE = "non_native"