# ManDi

## Overview
ManDi (Mandarin Chinese Dialect Corpus) is a spoken corpus of regional Mandarin dialects and Standard Mandarin.  The corpus currently contains a total of 357 recordings from 36 speakers of six Mandarin dialects.

The speakers recorded production of monosyllabic words, disyllabic words, short sentences, a short passage North Wind and the Sun and a Chinese modern poem Wo Chun, in Standard Mandarin and their own regional dialect--one of six regional Mandarin dialects, i.e. Beijing, Chengdu, Jinan, Taiyuan, Wuhan, and Xi’an Mandarin.

The corpus was collected remotely using participant-controlled smartphone recording apps. Word- and phone-level alignments were generated using Praat and the Montreal Forced Aligner.



```bibtex
@inproceedings{zhao2022mandi,
  title={The ManDi corpus: A spoken corpus of Mandarin regional dialects},
  author={Zhao, Liang and Chodroff, Eleanor},
  booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages={1985--1990},
  year={2022}
}
```

## Instructions

Define the path where **ManDi** will be stored

```bash
export H2T_DATADIR=""
```

Run the Python script to generate the processed data:

```bash
python generate.py
```

## Expected Output

After running the steps above, your directory layout will be:

```
${H2T_DATADIR}/
└─ mandi/
   └─ audio/
      └─ zh/
      │  ├─ 000.wav
      │  ├─ 002.wav
      │  └─ ...
      └─ ...
```

If your generate.py script writes manifests,, you should get JSONL files for the zh-?? language pair under your chosen output path (e.g., ./manifest/mandi/). A jsonl entry looks like:

```json
{
 	"dataset_id": "mandi",
	"sample_id": "000",
	"src_audio": "/mandi/audio/zh/000.wav",
	"src_ref": "一会儿 他们俩就商量好了 那个人马上就把袍子脱了下来 到了末了 北风就卯足了劲儿 拼命的吹 就算他的本领大 北风跟太阳正在那争论谁的本领大 所以北风不得不承认 可是 只好就算了 说 谁能先叫这个过路的把他的袍子脱下来 有一回 他吹的越厉害 太阳出来一晒 来了一个过路的 身上穿了一件厚袍子 那个人就把他的袍子裹得越紧 北风没辙了 还是太阳比他的本领大 说着说着",
	"tgt_ref": null,
	"src_lang": "zh",
	"tgt_lang": "en",
	"benchmark_metadata": {
	"native_acc": "BEI",
	"spoken_acc": "BEI",
	"participant_id": "004",
	"context": "short"
}
```

## License

For non-commercial use under a CC BY-NC 3.0 license.
