## Installation

### Requirements

To run the evaluation scripts and download the models, ensure you have the following Python libraries installed.

```bash
pip install fasttext
pip install lingua-language-detector
pip install unbabel-comet
pip install sacrebleu
pip install datasets
pip install huggingface_hub
```

### Set environment variables

Before running any scripts, make sure to configure your Hugging Face cache paths and authentication token.

```bash
export HUGGINGFACE_TOKEN=''          # your personal HF token
export HF_HOME="/path/to/hf_cache"   # base directory for HF cache
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DATASETS_CACHE="$HF_HOME/datasets"
```

---

### Handling Unbabel Models

When using `Unbabel` models (e.g., `Unbabel/wmt22-cometkiwi-da`, `Unbabel/XCOMET-XL`, `Unbabel/XCOMET-XXL`), the code will automatically attempt to download them from Hugging Face.

⚠️ **Important**: Many of these models require requesting access on Hugging Face. If you try to load them without an access token, the download will fail.

To avoid this:

1. Request access to the model from its Hugging Face page.
2. Once access is granted, log in to Hugging Face with your token:

```bash
huggingface-cli login
```

or, if you prefer environment variables:

```bash
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Download XComet-XXL model

```python
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = download_model("Unbabel/XCOMET-XXL")
print("Unbabel/XCOMET-XXL model path:", model_path)

# download XLM-Roberta-XXL model in HF CACHE
tokenizer = AutoTokenizer.from_pretrained('facebook/xlm-roberta-xxl')
model = AutoModelForMaskedLM.from_pretrained("facebook/xlm-roberta-xxl")
```

---

### Download and Use GlotLID

The [GlotLID model](https://huggingface.co/cis-lmu/glotlid) is distributed via Hugging Face but needs to be loaded with `fasttext`.

```python
import fasttext
from huggingface_hub import hf_hub_download

model_path = hf_hub_download( repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None )
print("GlotLID model path:", model_path)
```

---

### Download MetricX24-XXL Model

The [MetricX-24 models](https://huggingface.co/google/metricx-24-hybrid-large-v2p6) are hosted by Google on Hugging Face. They require both a checkpoint and a tokenizer (usually `mt5-xl`-based).

```python
from huggingface_hub import snapshot_download

model_name = "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
model_path = snapshot_download(repo_id=model_name)
print("MetricX-24-XXL model is downloaded to:", model_path)
```

```python
from huggingface_hub import snapshot_download

model_name = "google/mt5-xxl"
model_path = snapshot_download(repo_id=model_name, ignore_patterns=['tf_model.h5', 'pytorch_model.bin'])

print("T5 tokenizer model is downloaded to:", model_path)
```

### Set Model Paths as env variables

After downloading the necessary models, you need to set their paths as environment variables. All models are stored in $HF_HOME cache folder, for instance:

```bash
export METRICX_CK_NAME='${HF_HOME}/hub/models--google--metricx-24-hybrid-xxl-v2p6/snapshots/0ff238ccb517eb0b2998dd6d299528b040c5caec' 
export METRICX_TOKENIZER='${HF_HOME}/hub/models--google--mt5-xxl/snapshots/e07c395916dfbc315d4e5e48b4a54a1e8821b5c0'
export XCOMET_CK_NAME='${HF_HOME}/hub/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt'
export GlotLID_PATH='${HF_HOME}/hub/models--cis-lmu--glotlid/snapshots/74cb50b709c9eefe0f790030c6c95c461b4e3b77/model.bin'
```