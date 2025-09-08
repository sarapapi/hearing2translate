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
    dataset_id = "europarl_st"

    dataset_path = Path(__file__).parent 
    url = "https://www.mllp.upv.es/europarl-st/v1.1.tar.gz"
    filename = dataset_path / Path("v1.1.tar.gz")
    extract_dir = dataset_path / Path("europarl-st-v1.1")

    # Download if not already present
    #if not filename.exists() and not extract_dir.exists():
    #    print("Downloading...")
    #    with TqdmUpTo(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = str(filename)) as t:
    #        urllib.request.urlretrieve(url, filename, reporthook= t.update_to)

    ## Extract if not already extracted
    #if not os.path.exists(extract_dir):
    #    print("Extracting...")
    #    with tarfile.open(filename, "r:gz") as tar:
    #        tar.extractall()
    #    print("Done.")
    #else:
    #    print("Already downloaded and extracted Europarl-ST")

    extract_dir = Path("/scratch/translectures/data/Europarl-ST/RELEASES/v1.1/")
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

                df_full_src = df.groupby(["audio"]).apply(lambda x: " ".join(list(x["src"])), include_groups=False)
                df_full_tgt = df.groupby(["audio"]).apply(lambda x: " ".join(list(x["tgt"])), include_groups=False)

                with jsonlines.open(dataset_path/f"{src}-{tgt}.jsonl", mode="w") as writer:
                    samples = []
                    for i, (doc, src_ref, tgt_ref) in enumerate(zip(file_list, df_full_src, df_full_tgt, strict=True)):
                        #sample_path = (Path(__file__).parent / "audio" / src / doc).with_suffix(".wav")
                        sample_path_json = Path(dataset_id) / "audio" / src / f"{doc}.wav"

                        samples.append(
                                asdict(InputJson(
                                    dataset_id=dataset_id,
                                    sample_id=i,
                                    src_audio=str(sample_path_json),
                                    src_ref=src_ref,
                                    tgt_ref=tgt_ref,
                                    src_lang= src,
                                    ref_lang= tgt,
                                    benchmark_metadata={"context" : "long", "doc_id" : doc, "dataset_type" : DatasetType.LONGFORM }
                                ))
                                )
                        #if not sample_path.is_file():
                        #    sample_path.symlink_to( audios.absolute() / f"{doc}.m4a")
                    writer.write_all(samples)

            print("Tranforming m4a to wav files...")
            #for f, _ in tqdm(df_full_tgt.items()):
            for wav_f in tqdm(file_list):
                #Using pydub to read m4a
                audio_path = dataset_path / "audio" / src / f"{wav_f}.wav"
                if audio_path.is_file():
                    continue
                audio = AudioSegment.from_file(audios.absolute() / f"{wav_f}.m4a")
                audio = audio.set_frame_rate(16000)
                audio = audio.set_channels(1)
                audio.export(audio_path, format="wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download')
    args = parser.parse_args()
    generate_europarl_st()