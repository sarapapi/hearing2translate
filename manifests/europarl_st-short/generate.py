import os
from dataclasses import asdict
from pathlib import Path
import urllib.request
from tqdm import tqdm

import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_schema import InputJson, DatasetType

import jsonlines
from datasets import load_dataset, Dataset
from langcodes import Language
from dotenv import load_dotenv
import pandas as pd 
import os
import tarfile
import urllib.request
import csv
from tqdm import tqdm
import argparse
import soundfile as sf
import librosa
from pydub import AudioSegment

class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

load_dotenv()
langs = {"en", "es", "fr", "de", "it", "pt" }
not_used = {"pl", "ro", "nl"}

def generate_europarl_st():
    print("Generating europarl_st dataset")
    dataset_id = "europarl_st-short"

    dataset_path = Path(__file__).parent 
    extract_dir = os.getenv('EUROPARL_ST_PATH')
    if not extract_dir:
        raise Exception("Please, set the environment variable EUROPARL_ST_PATH to the path of the root directory of the downloaded dataset.")
    else:
        extract_dir = Path(extract_dir)

    #extract_dir = Path("/scratch/translectures/data/Europarl-ST/RELEASES/v1.1/")
    for src in langs:
        for tgt in langs - {src}:
            audios = extract_dir / f"{src}" / "audios"
            (Path(__file__).parent / "audio" / src).mkdir(parents=True,exist_ok=True)
            for split in ("test",): #dev
                f = extract_dir / f"{src}/{tgt}/{split}"
                print(f"Processing Europarl-ST: {split}|{src}-{tgt}")
                df = pd.read_csv(f/"segments.lst", delimiter=" ", names=["audio","s","e"])
                base= (f /"segments")
                with open(base.with_suffix(f".{src}"), "r") as src_f, open(base.with_suffix(f".{tgt}"), "r") as tgt_f:
                    df["src"] = pd.Series(map(str.rstrip, src_f.readlines()))
                    df["tgt"] = pd.Series(map(str.rstrip, tgt_f.readlines()))

                #df["audio"] = df["audio"].transform(get_audio_path)
                file_list = df["audio"].unique()

                with jsonlines.open(dataset_path/f"{src}-{tgt}.jsonl", mode="w") as writer:
                    samples = []
                    for i, audio, s, e, src_ref, tgt_ref in tqdm(df.itertuples()):
                        sample_path_json = Path(dataset_id) / "audio" / src / f"{audio}_{s}_{e}.wav"

                        samples.append(
                                asdict(InputJson(
                                    dataset_id="europarl_st",
                                    sample_id=i,
                                    src_audio=str(sample_path_json),
                                    src_ref=src_ref,
                                    tgt_ref=tgt_ref,
                                    src_lang= src,
                                    ref_lang= tgt,
                                    benchmark_metadata={"context" : "short", "doc_id" : audio, "dataset_type" : DatasetType.LONGFORM }
                                ))
                                )
                        audio_path = dataset_path / "audio" / src / f"{audio}_{s}_{e}.wav"
                        if audio_path.is_file():
                            continue
                        y, sr = librosa.load(audios.absolute() / f"{audio}.m4a", sr=16_000, mono=True, offset=s, duration=e-s)
                        sf.write(audio_path, y, samplerate=16_000)
                    writer.write_all(samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download')
    args = parser.parse_args()
    generate_europarl_st()