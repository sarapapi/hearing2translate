# UCLASS RELEASE TWO

## Overview

University College London's Archive of Shuttered Speech (UCLASS) Release Two is a recording of speakers who shutter in a normal speaking condition. The dataset contians spontaneous dysfluent monologs, read passages and conversations in English language.

```bibtex
@article{howell2009university,
  title={The university college london archive of stuttered speech (uclass)},
  author={Howell, Peter and Davis, Stephen and Bartrip, Jon},
  journal={Journal of speech, language, and hearing research},
  volume={52},
  number={2},
  pages={556--569},
  year={2009}
}
```

## Instructions

Define the path where **UCLASS2** will be stored:

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
└─ uclass2/
   └─ audio/
	└─ en/
      └─ Monolog/
      │  ├─ F_0142_11y3m_1.wav
      │  ├─ F_0142_11y5m_1.wav
      │  └─ .../
      └─ Reading/
      │  ├─ F_0101_14y8m_1.wav
      │  ├─ F_0101_14y8m_2.wav
      │  └─ ...
      └─ Conversation/
      │  ├─ F_0101_10y4m_1 .wav
      │  ├─ F_0101_13y1m_1.wav
      │  └─ ...
      └─ ...
```

If your generate.py script writes manifests, you should get JSONL files (one per context and language pair) under your chosen output path (e.g., ./manifests/uclass2/monolog). A jsonl entry looks like:


```json
			{	"dataset_id": "uclass2",
				"sample_id": "01",
				"src_audio": "/uclass2/audio/en/monolog/F_0142_11y3m_1.wav",
				"src_ref": null,
				"tgt_ref": null,
				"src_lang": "en",
				"tgt_lang":	"de",
				"benchmark_metadata": {
						"native_acc": null,
						"spoken_acc": null,
						"participant_id": null,
						"context": "monolog",
			}
```



## License

All datasets are available for non-commercial research
