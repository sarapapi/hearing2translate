
# Europarl-ST

## Overview

Europarl-ST is a multilingual speech translation dataset based on European Parliament interventions.

* **Primary Task:** General benchmarking of parliamentary interventions.
* **Secondary Task:** Other speech translation experiments.
* Supports **short-form (sentence-level)** usage and **document-level** evaluation, though **only the short-form version** is provided here.
* Average intervention length: \~1 minute 40 seconds (varies by language direction).

Supported language directions:

* `en → {es, fr, pt, it, de, nl}`
* `{es, fr, pt, it, de} → en`
* `en → zh with no tgt based on the en-de split is also generated for easy non-reference based evaluation`

More information:

* [Official website](https://www.mllp.upv.es/europarl-st/)
* [IEEE article](https://ieeexplore.ieee.org/document/9054626)
* [arXiv paper](https://arxiv.org/abs/1911.03167)

```bibtex
@INPROCEEDINGS{9054626,
  author={Iranzo-Sánchez, Javier and Silvestre-Cerdà, Joan Albert and Jorge, Javier and Roselló, Nahuel and Giménez, Adrià and Sanchis, Albert and Civera, Jorge and Juan, Alfons},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Europarl-ST: A Multilingual Corpus for Speech Translation of Parliamentary Debates}, 
  year={2020},
  volume={},
  number={},
  pages={8229-8233},
  keywords={Training;Adaptation models;Filtering;Pipelines;Europe;Task analysis;Speech processing;speech translation;spoken language translation;automatic speech recognition;machine translation;multilingual corpus},
  doi={10.1109/ICASSP40776.2020.9054626}}
```

---

## Instructions

1. Run the `generate.py` script to download and prepare the dataset:

   ```bash
   python generate.py
   ```

2. The script will:

   * Download the full dataset (\~20 GB, including training data).
   * Resample audio to 16 Hz.
   * Convert audio to `.wav` format for easier downstream processing and segmentantion

3. If you already have the dataset downloaded, set the environment variable `EUROPARL_ST_PATH` to the root folder:

   ```bash
   export EUROPARL_ST_PATH=/path/to/europarl-st
   ```

---

## Expected Output

* A fully prepared Europarl-ST dataset in `.wav` format at 16 Hz uder the `europarl_st/audio` folder
* Directory structure compatible with supported language directions (`en → {es, fr, pt, it, de}` and reverse).

---

## License

The Europarl-ST dataset is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
