# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import wget
import json
random.seed(42)
from datasets import Dataset, DatasetDict, Audio

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def download_file(url, download_path):
    filename = url.rpartition("/")[-1]
    if not (download_path / filename).exists():
        try:
            # download file
            print(f"Downloading {filename} from {url}")
            custom_bar = (
                lambda current, total, width=80: wget.bar_adaptive(
                    round(current / 1024 / 1024, 2),
                    round(total / 1024 / 1024, 2),
                    width,
                )
                + " MB"
            )
            wget.download(url, out=str(download_path / filename), bar=custom_bar)
        except Exception as e:
            message = f"Downloading {filename} from {url} failed! ({e})"
            raise HTTPError(url, None, message, None, None)
    return True


def extract_tgz(tgz_filepath, extract_path, out_filename=None):
    if not tgz_filepath.exists():
        raise FileNotFoundError(f"{tgz_filepath} is not found!!")
    tgz_filename = tgz_filepath.name
    tgz_object = tarfile.open(tgz_filepath)
    if not out_filename:
        out_filename = tgz_object.getnames()[0]
    # check if file is already extracted
    if not (extract_path / out_filename).exists():
        for mem in tqdm(tgz_object.getmembers(), desc=f"Extracting {tgz_filename}"):
            out_filepath = extract_path / mem.get_info()["name"]
            if mem.isfile() and not out_filepath.exists():
                tgz_object.extract(mem, path=extract_path)
    tgz_object.close()


def download_extract_file_if_not(url, tgz_filepath, download_filename):
    download_path = tgz_filepath.parent
    if not tgz_filepath.exists():
        # download file
        download_file(url, download_path)
    # extract file
    extract_tgz(tgz_filepath, download_path, download_filename)


def load_noise_samples(noise_path):
    download_extract_file_if_not(
        url="https://dl.fbaipublicfiles.com/muavic/noise_samples.tgz",
        tgz_filepath=(noise_path/"noise_samples.tgz"),
        download_filename="noise_samples"
    )
    noise_dict = defaultdict(list)
    for wav_filepath in (noise_path/"noise_samples").rglob('*.wav'):
        category = wav_filepath.parent.stem
        noise_dict[category].append(str(wav_filepath))
    return noise_dict


def add_noise(signal, noise, snr):
    """
    signal: 1D tensor in [-32768, 32767] (16-bit depth)
    noise: 1D tensor in [-32768, 32767] (16-bit depth)
    snr: tuple or float
    """
    signal = signal.astype(np.float32)
    noise = noise.astype(np.float32)

    if type(snr) == tuple:
        assert len(snr) == 2
        snr = np.random.uniform(snr[0], snr[1])
    else:
        snr = float(snr)

    if len(signal) > len(noise):
        ratio = int(np.ceil(len(signal) / len(noise)))
        noise = np.concatenate([noise for _ in range(ratio)])
    if len(signal) < len(noise):
        start = 0
        noise = noise[start : start + len(signal)]

    amp_s = np.sqrt(np.mean(np.square(signal), axis=-1))
    amp_n = np.sqrt(np.mean(np.square(noise), axis=-1))
    noise = noise * (amp_s / amp_n) / (10 ** (snr / 20))
    mixed = signal + noise

    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed
 

def mix_audio_with_noise(audio_file, out_file, noise_wav_file, snr):
    # read audio WAV file
    sr, audio = wavfile.read(audio_file)
    # read noise WAV file
    _, noise_wav = wavfile.read(noise_wav_file)
    # mix audio + noise
    mixed = add_noise(audio, noise_wav, snr)
    # save resulting noisy audio WAV file
    wavfile.write(out_file, sr, mixed)
    return mixed

def collect_ambient_wavs(musan_path: Path):
    """
    Collect all .wav files from music/ and noise/ subfolders,
    and balance the number of music and noise files.
    """
    music_dir = musan_path / "music"
    noise_dir = musan_path / "noise"

    # recursively find all .wav files
    music_wavs = list(music_dir.rglob("*.wav"))
    noise_wavs = list(noise_dir.rglob("*.wav"))

    # balance the number of files
    min_len = min(len(music_wavs), len(noise_wavs))
    music_wavs = random.sample(music_wavs, min_len)
    noise_wavs = random.sample(noise_wavs, min_len)

    # merge both lists
    ambient_wavs = music_wavs + noise_wavs
    random.shuffle(ambient_wavs)

    return ambient_wavs


def main(args):
    dataset_id =  f"noisy_fleurs_{args.noise_type}"
    fleurs_path = args.fleurs_manifest_dir
    noise_path = args.h2t_datadir / f"{dataset_id}_wavs"
    manifest_dir = args.manifest_dir / dataset_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    noise_path.mkdir(parents=True, exist_ok=True)

    # start loading resources
    logger.info("Loading noise samples...")
    if args.noise_type == "babble":
        noise  = load_noise_samples(args.h2t_datadir)
        noise_wav_files = noise["party"]
    elif args.noise_type == "ambient":
        noise_wav_files = collect_ambient_wavs(args.musan_path)
    else:
        ValueError("Only noise type babble and ambient are supported.")

    noise_snr = 0  # as in MuAViC paper
    all_samples = [] 
    # loop over all JSONL files in fleurs_path
    for jsonl_file in tqdm(list(fleurs_path.glob("*.jsonl")), desc="Processing manifests"):
        logger.info(f"Processing {jsonl_file.name}...")
        new_jsonl_lines = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                src_audio_path =  args.h2t_datadir / sample["src_audio"].lstrip("/")
                src_lang = sample["src_lang"]
                
                # create output subdir for this language
                lang_outdir = noise_path / src_lang
                lang_outdir.mkdir(parents=True, exist_ok=True)
                noisy_audio_filename = src_audio_path.name.replace(".wav", f"_{args.noise_type}.wav")
                noisy_audio_path = lang_outdir / noisy_audio_filename

                # pick a random noise file
                noise_wav_file = noise_wav_files[random.randint(0, len(noise_wav_files)-1)]

                # mix audio with noise
                mix_audio_with_noise(
                    src_audio_path, noisy_audio_path,
                    noise_wav_file, noise_snr
                )

                # update JSON with new noisy audio path
                sample["dataset_id"] = dataset_id
                sample["src_audio"] = f"{dataset_id}_wavs/{src_lang}/{noisy_audio_filename}"
                all_samples.append(sample)
                new_jsonl_lines.append(json.dumps(sample, ensure_ascii=False))

        # save new JSONL manifest
        out_jsonl_path = manifest_dir / jsonl_file.name
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_jsonl_lines))

        logger.info(f"Saved noisy manifest to {out_jsonl_path}")

    logger.info("Building Hugging Face Dataset...")
    for sample in all_samples:
        sample["src_audio"] = str(args.h2t_datadir / sample["src_audio"])
    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_list(all_samples)

    # Cast audio column to Audio type (HF will load waveforms automatically)
    hf_dataset = hf_dataset.cast_column("src_audio", Audio())

    # Create a DatasetDict (use test split since you only have test)
    hf_dataset_dict = DatasetDict({"test": hf_dataset})

    # Save to disk in Hugging Face format
    out_hf_dir = manifest_dir / f"{dataset_id}_hf"
    hf_dataset_dict.save_to_disk(out_hf_dir)


    logger.info("All files processed!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h2t-datadir",
        type=Path,
        required=True,
        help="Path to H2T dataset directory (used for audio or metadata)"
    )

    parser.add_argument(
        "--fleurs-manifest-dir", type=Path, required=True,
        help="Relative/Absolute path where FLEURS manifest is stored."
    )

    parser.add_argument(
        "--manifest-dir", type=Path, required=True,
        help="Relative/Absolute path to directory of the manifests where the noisy version of FLEURS will be stored."
    )

    parser.add_argument(
        "--musan-path",
        type=Path,
        required=True,
        help="Path to MUSAN dataset (speech, music, noise) used for adding noise"
    )


    parser.add_argument(
        "--noise-type",
        type=str,
        required=True,
        choices=["babble", "ambient"],
        help="Type of noise to add: 'babble' for multiple speakers, 'ambient' for environmental sounds (music, car, siren, etc.)"
    )

    args = parser.parse_args()
    main(args)
