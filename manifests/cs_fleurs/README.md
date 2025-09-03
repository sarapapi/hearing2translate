# How to use CS-FLEURS (Codeswitched FLEURS)

# Dependencies
Need datasets<=3.6.0 due to using a Huggignface dataset script (is depricated)

# Description
The `generate.py` script downloads and prepares the 4 langs supported by the used models from splits of the CS-FLEURS-READ subset of CS-FLEURS, a code-switeched version of the FLEURS dataset. 

From the article
> "...for each code-switched pair, one language, referred to as Matrix,
provides grammatical structure while a second language, re-
ferred to as Embedded, provides words or morphological units
that are inserted within the sentence. We refer to language pairs
as Matrix-Embedded; for instance, Mandarin-English refers to
a Mandarin matrix sentence with English words embedded..."

For all our languages, **English** is the embedded langaugue and **target** langauge.

```bash
python generate.py 
```
# Languages directions
- de-en
- es-en
- fr-en
- zh-en

# Reference and Links
https://www.isca-archive.org/interspeech_2025/yan25c_interspeech.html
https://huggingface.co/datasets/byan/cs-fleurs