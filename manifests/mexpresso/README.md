
# mExpresso

## Overview

mExpresso is an expressive S2ST dataset that includes seven styles of read speech (i.e., default, happy, sad, confused, enunciated, whisper and laughing) between English and five other languages.
We used the released speech-to-text split.

* **Primary Task:** Benchmarking model performance for audios with high degrees of emotion.
* Short segments

Supported language directions:

* `en → {es, fr, it, de, zh}`

More information:

* [Seamless communication mExpresso dataset info](https://github.com/facebookresearch/seamless_communication/tree/main/docs/expressive#mexpresso-multilingual-expresso)
```bibtex
@inproceedings{seamless2023,
   title="Seamless: Multilingual Expressive and Streaming Speech Translation",
   author="{Seamless Communication}, Lo{\"i}c Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-juss{\`a}, Maha Elbayad, Hongyu Gong, Francisco Guzm{\'a}n, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, Mary Williamson",
  journal={ArXiv},
  year={2023}
}
```
---

## Instructions

1. Run the `generate.py` script to download and prepare the dataset:

   ```bash
   python generate.py
   ```

2. The script will:

   * Download the a mirror of the dataset upluaded to HuggingFace.
   * Create the audio dataset, linking the audio files from the HuggingFace cache.
   * Generate the .jsonl for each language pair.
---

## Expected Output

* A fully prepared dataset in the `mexpresso/audio` folder and .jsonl files for each language pair.

---

## License

The mExpresso dataset is derived from the Expresso dataset. Text only part of the mExpresso dataset is licensed under MIT, while the original Expresso audios are under a Creative Commons Attribution Non Commercial 4.0 (cc-by-nc-4.0)

