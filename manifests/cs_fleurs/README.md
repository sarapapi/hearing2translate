# CS-FLEURS (Codeswitched FLEURS)

## Overview

CS-FLEURS is a **code-switched** version of the FLEURS dataset designed for:

* **Primary Task:** Code-switching research.
* **Secondary Task:** Gender-bias analysis.

This is a **short-form dataset** derived from the **CS-FLEURS-READ-TEST** subset, which supports **14 directions in total** (`XXX-eng`). In this release, **4 language directions** are provided.

From the original article:

> "...for each code-switched pair, one language, referred to as Matrix,
> provides grammatical structure while a second language, referred to as Embedded, provides words or morphological units
> that are inserted within the sentence. We refer to language pairs
> as Matrix-Embedded; for instance, Mandarin-English refers to
> a Mandarin matrix sentence with English words embedded..."

For all supported directions here, **English** is both the **embedded** and **target** language.

* [Interspeech 2025 Paper](https://www.isca-archive.org/interspeech_2025/yan25c_interspeech.html)
* [Hugging Face Dataset](https://huggingface.co/datasets/byan/cs-fleurs)
```bibtex
@inproceedings{yan25c_interspeech,
  title     = {{CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset}},
  author    = {Brian Yan and Injy Hamed and Shuichiro Shimizu and Vasista Sai Lodagala and William Chen and Olga Iakovenko and Bashar Talafha and Amir Hussein and Alexander Polok and Kalvin Chang and Dominik Klement and Sara Althubaiti and Puyuan Peng and Matthew Wiesner and Thamar Solorio and Ahmed Ali and Sanjeev Khudanpur and Shinji Watanabe},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {743--747},
  doi       = {10.21437/Interspeech.2025-2247},
  issn      = {2958-1796},
}
```
---

## Instructions

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the `generate.py` script to download and prepare the dataset:

   ```bash
   python generate.py
   ```
3. The script will:

   * Download and prepare the **4 supported languages** from the **CS-FLEURS-READ** subset.
   * Link the audio files from the Hugging Face cache.

---

## Expected Output

* A processed CS-FLEURS dataset containing the 4 supported code-switched language directions, ready for use in code-switching and gender-bias tasks.

Supported directions:

* `de-en`
* `es-en`
* `fr-en`
* `zh-en`

---

## License

CS-Fleurs is released under a Creative Commons Attribution Non Commercial 4.0 license (cc-by-nc-4.0).
