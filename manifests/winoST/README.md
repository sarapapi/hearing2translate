

wget https://zenodo.org/records/4139080/files/WinoST.zip?download=1 -O ./WinoST.zip
unzip -d extracted WinoST.zip && find extracted -name "*.wav.gz" -exec gunzip {} \;

wget https://github.com/gabrielStanovsky/mt_gender/raw/refs/heads/master/data/aggregates/en.txt

then run python generate.py

# WinoST

This repository provides scripts for downloading and preparing the **WinoST** dataset.

## Download the dataset

Download and unzip the **WinoST** dataset from Zenodo:

```bash
wget https://zenodo.org/records/4139080/files/WinoST.zip?download=1 -O ./WinoST.zip
unzip -d extracted WinoST.zip
````

Decompress the audio files (`.wav.gz`):

```bash
find extracted -name "*.wav.gz" -exec gunzip {} \;
```

## Download WINO dataset sentences

Download the aggregated English references from the **mt\_gender** repository:

```bash
wget https://github.com/gabrielStanovsky/mt_gender/raw/refs/heads/master/data/aggregates/en.txt
```

## Generate Input Jsons

Run the Python script to generate the processed data:

```bash
python generate.py
```