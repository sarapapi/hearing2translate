<p align="center">
  <img src="assets/logo_h2t.png" alt="Hearing to Translate" width="420"/>
</p>

<h2 align="center">
  Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
</h2>

The **Hearing-to-Translate** test suite provides a unified evaluation framework for 
assessing how effectively SpeechLLMs, Speech Foundation Models (SFMs), and cascaded 
ASR→LLM pipelines handle speech-to-text translation across diverse real-world conditions. 
Covering 22 systems, 13 language pairs, 9 speech phenomena, and 16 benchmarks, the suite 
measures performance on clean speech as well as challenging scenarios involving gender 
bias, accents, code-switching, disfluencies, noise, named entities, emotion, and 
long-form content.

---

## 📰 News
- Mar. 25, 2026: added support for the [Gemini 2.5 Flash model](inference/api/gemini.py) and their [outputs](outputs/gemini-2.5-flash) for generic benchmarks.
- Feb. 10, 2026: added support for the [Qwen3-Omni model](inference/speechllm/qwen3omni.py) and their [outputs](outputs/qwen3omni)
- Dec. 28, 2025: [Human Evaluation data released on 🤗HuggingFace](https://huggingface.co/datasets/zouharvi/hearing2translate-humeval)
- Dec. 19, 2025: [Preprint released on arXiv](https://arxiv.org/abs/2512.16378)

## Repository Structure
```
.
├── analysis/ # Scripts and files used for the analysis and aggregation of metrics
├── evaluation/ # Evaluation scripts (XCOMET, MetricX, LID)
├── evaluation_human/ # Code and data of human evaluation
├── inference/ # Generation scripts for each model 
├── manifests/ # Code for using and replicating the manifests
├── outputs/ # Outputs produced by the models
├── infer.py # Inference script
├── run_text_models.sh # Example script for running text-based models
├── infer-loop.sh # Script for repeated / batch inference
├── requirements.txt # Python dependencies
└── README.md
```

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/sarapapi/hearing2translate.git
cd hearing2translate
pip install -r requirements.txt
```
**Note:** The required `transformers` version depends on the specific model being used. 
Please ensure that you install the version compatible with the model you intend to run, as reported in [Table 6 of the paper](https://arxiv.org/abs/2512.16378).

## Usage

### 1. Download benchmarks and set data directory
Download the desired benchmarks by following the instructions provided in each benchmark-specific README, 
and set `${H2T_DATADIR}` to the directory containing the corresponding audio files.

**Supported benchmarks by category:**
- **Generic**: [`fleurs`](manifests/fleurs/README.md), [`covost2`](manifests/covost2/README.md), [`europarl_st`](manifests/europarl_st/README.md), [`wmt`](manifests/wmt/README.md)
- **Gender Bias**: [`winoST`](manifests/winoST/README.md)
- **Accents**: [`commonAccent`](manifests/commonAccent/README.md), [`mandi`](manifests/mandi/README.md)
- **Code Switching**: [`cs-dialogue`](manifests/cs-dialogue/README.md), [`cs_fleurs`](manifests/cs_fleurs/README.md)
- **Disfluencies**: [`libristutter`](manifests/libristutter/README.md)
- **Noise**: [`noisy_fleurs_ambient`](manifests/noisy_fleurs_ambient/README.md), [`noisy_fleurs_babble`](manifests/noisy_fleurs_babble/README.md)
- **Emotion**: [`emotiontalk`](manifests/emotiontalk/README.md), [`mexpresso`](manifests/mexpresso/README.md)
- **Long-Form**: [`acl6060-long`](manifests/acl6060-long/README.md), [`acl6060-short`](manifests/acl6060-short/README.md), [`mcif-long`](manifests/mcif-long/README.md), [`mcif-short`](manifests/mcif-short/README.md)

To use the Gemini API, you must set the environment variable with your API key:
```
export GEMINI_API_KEY=<your-api-key>
```

### 2. Run inference

Run inference with the following command:
  ```python
  python infer.py \
    --model ${MODEL_NAME} \
    --in-modality {speech/text} \
    --in-file ./manifests/${BENCHMARK_NAME}/${SRC_LANG}-${TGT-LANG}.jsonl \
    --out-file ${OUTPUT_PATH}
  ```
The full list of supported models can be obtained with `python infer.py -h`.
Supported benchmarks are listed above, while benchmark-specific language coverage is documented in the corresponding READMEs.

### 3. Run evaluation

After generating model outputs, run the evaluation suite using the scripts in the `evaluation/` directory.
For environment setup, model downloads, and benchmark-specific evaluation commands, refer to the dedicated [Evaluation README](evaluation/README.md).

## Contributing

If you want to add a model to the repository, please create a PR with:
- the inference code in `inference/{llm/sfm/speechllm}`
- the outputs on all the applicable benchmarks of the test suite in `outputs/${MODEL_NAME}`

Please refer to [PR template](pull_request_template.md) for more information.

## License

The code contained in this repository is released under [Apache 2.0 License](#license). 

Benchmarks are released under their own licenses. See the specific READMEs in `/manifests` for more information.

Human evaluation data in `evaluation_human/hearing2translate-v1` is released under CC-BY 4.0 License.

## Citation

```bibtex
@misc{papi2025hearingtranslateeffectivenessspeech,
      title={Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs}, 
      author={Sara Papi and Javier Garcia Gilabert and Zachary Hopton and Vilém Zouhar and Carlos Escolano and Gerard I. Gállego and Jorge Iranzo-Sánchez and Ahrii Kim and Dominik Macháček and Patricia Schmidtova and Maike Züfle},
      year={2025},
      eprint={2512.16378},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.16378}, 
}
```
