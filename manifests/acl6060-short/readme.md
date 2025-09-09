## Usage
1.	Set the environment variable
Make sure to set H2T_DATADIR.
    
2.	Generate manifests
Run the script to create the JSONL files.

- On the first run, the files will be created.
- On subsequent runs, new records will be appended.

```
python manifests/acl6060-short/generate.py
```

3.	Run inference
Use the generated JSONL files as input for your model inference.