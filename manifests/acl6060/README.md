# ACL-6060

The short- and long- form of the dataset are obtained independently from a different source:

- **Hugging Face dataset**: [`ymoslem/acl-6060`](https://huggingface.co/datasets/ymoslem/acl-6060) (split: `eval`)  
  
- **Manually downloaded**: [ACL Anthology](https://aclanthology.org/2023.iwslt-1.2)  
  `manifests/acl6060/long/<lang>.xml`
  
--

## What to do
1. Set `H2T_DATADIR` and run the script.
```
python manifests/acl6060/generate.py
```

2. Download additional long-form audio files from [GitHub release](https://github.com/sarapapi/hearing2translate/releases/tag/data-share-acl6060), and place them under `$H2T_DATADIR/manifests/acl/en`.


3. Run inference of your model.
