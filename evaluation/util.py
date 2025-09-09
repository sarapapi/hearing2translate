import numpy as np
from metrics.comet.metric import BaseCOMET
from metrics.comet_kiwi.metric import COMETKiwi
from metrics.xcomet.metric import XCOMET, XCOMET_QE
from metrics.metricx.metric import RefMetricX_24, QEMetricX_24
import sacrebleu
import json
import fasttext

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# note that downloading Comet models requires HF token and gated access
COMET_CK_NAME='Unbabel/wmt22-comet-da'
COMET_KIWI_CK_NAME='Unbabel/wmt22-cometkiwi-da'
XCOMET_CK_NAME='Unbabel/XCOMET-XL' #Unbabel/XCOMET-XL

METRICX_CK_NAME='google/metricx-24-hybrid-large-v2p6'
METRICX_TOKENIZER='google/mt5-xl'

# we need to downlad the fasttext model
GlotLID_PATH=''


MAPPING_TO_FASTTEXT_LABEL = {
    'it': 'ita_Latn', 'es':'spa_Latn', 'de':'deu_Latn', 'zh':'cmn', 
    'pt': 'por_Latn', 'en':'eng_Latn', 'fr':'fra_Latn', 'nl': 'nld_Latn'
}


@dataclass
class InputJson:
    """Schema for each entry in the input jsonl file."""
    dataset_id: str
    sample_id: int
    src_audio: Optional[str]
    src_ref: Optional[str]
    tgt_ref: Optional[Dict[str, Any]]
    src_lang: str
    ref_lang: str
    benchmark_metadata: Optional[Dict[str, Any]]

@dataclass
class OutputJson:
    """Schema for each entry in the output jsonl file."""
    dataset_id: str
    sample_id: int
    src_lang: str
    tgt_lang: str
    output: str

@dataclass
class MergedData:
    """Schema for the combined data used in evaluation."""
    dataset_id: str
    sample_id: int
    src_lang: str
    tgt_lang: str
    model: str
    output: str
    src_ref: Optional[str]
    tgt_ref: Optional[Dict[str, Any]]
    src_audio: Optional[str]
    benchmark_metadata: Optional[Dict[str, Any]]


class Evaluator:
    """
    A class to handle the evaluation of model outputs against reference data.
    """

    def __init__(self, input_path: str, output_path: str, model_name: str):
        """
        Initializes the Evaluator.

        Args:
            input_path (str): Path to the input jsonl file with reference data.
            output_path (str): Path to the output jsonl file with model predictions.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.data: List[MergedData] = self._prepare_data()

    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads a JSONL file into a list of dictionaries."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON in {file_path}")
            return []
        return data

    def _prepare_data(self) -> List[MergedData]:
        """
        Loads data from input and output files and merges them based on
        their line index in the respective files.
        """
        input_data_raw = self._load_jsonl(self.input_path)
        output_data_raw = self._load_jsonl(self.output_path)

        if len(input_data_raw) != len(output_data_raw):
            print(
                f"Warning: Input file has {len(input_data_raw)} lines, but "
                f"output file has {len(output_data_raw)} lines. "
                "Merging will only proceed for the minimum length."
            )

        merged_data = []
        for i, (input_item, output_item) in enumerate(zip(input_data_raw, output_data_raw)):
            try:
                # Combine the data into our MergedData structure
                merged_entry = MergedData(
                    dataset_id=output_item['dataset_id'],
                    sample_id=output_item['sample_id'],
                    src_lang=output_item['src_lang'],
                    tgt_lang=output_item['tgt_lang'],
                    model=self.model_name,
                    output=output_item['output'],
                    src_ref=input_item.get('src_ref'),
                    tgt_ref=input_item.get('tgt_ref'),
                    src_audio=input_item.get('src_audio'),
                    benchmark_metadata=input_item.get('benchmark_metadata')
                )
                merged_data.append(merged_entry)
            except KeyError as e:
                print(f"Warning: Missing key {e} in data at line index {i}. Skipping this entry.")

        return merged_data

    def get_all_data(self) -> List[MergedData]:
        """Returns all merged data points as a list."""
        return self.data

    # --------------------------------------------------------------------------
    # Metric Evaluation Functions
    # --------------------------------------------------------------------------

    def evaluate_comet(self):
        """Evaluates the outputs using BaseCOMET."""
        self.comet = BaseCOMET(COMET_CK_NAME)
        batch_size = 8

        sources = [ item.src_ref for item in self.data]
        targets = [ item.tgt_ref for item in self.data]
        translations = [item.output for item in self.data]

        comet_result = self.comet.evaluate(translations, targets, sources, batch_size )
        return round(comet_result["system_score"], 4), comet_result["segments_scores"]

    def evaluate_comet_kiwi(self):
        """Evaluates the outputs using COMETKiwi (QE metric)."""
        self.comet_kiwi = COMETKiwi(COMET_KIWI_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        comet_kiwi_result = self.comet_kiwi.evaluate(translations, sources, batch_size)
        return round(comet_kiwi_result["system_score"], 4), comet_kiwi_result["segments_scores"]


    def evaluate_xcomet(self):
        """Evaluates the outputs using XCOMET."""
        self.xcomet = XCOMET(XCOMET_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        targets = [ item.tgt_ref for item in self.data]
        translations = [item.output for item in self.data]

        xcomet_result = self.xcomet.evaluate(translations, targets, sources, batch_size)
        return round(xcomet_result["system_score"], 4), xcomet_result["segments_scores"]


    def evaluate_xcomet_qe(self):
        """Evaluates the outputs using XCOMET_QE (QE metric)."""
        self.xcomet_qe = XCOMET_QE(XCOMET_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        xcomet_qe_result = self.xcomet_qe.evaluate(translations, [], sources, batch_size)
        return round(xcomet_qe_result["system_score"], 4), xcomet_qe_result["segments_scores"]


    def evaluate_ref_metricx(self):
        """Evaluates the outputs using RefMetricX_24."""
        self.ref_metricx = RefMetricX_24(METRICX_TOKENIZER, METRICX_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        targets = [item.tgt_ref for item in self.data]
        translations = [item.output for item in self.data]

        ref_metricx_result = self.ref_metricx.evaluate(
            hypotheses=translations, references=targets, sources=sources
        )
        return round(ref_metricx_result["system_score"], 4), ref_metricx_result["segments_scores"]


    def evaluate_qe_metricx(self):
        """Evaluates the outputs using QEMetricX_24 (QE metric)."""
        self.qe_metricx = QEMetricX_24(METRICX_TOKENIZER, METRICX_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        qe_metricx_result = self.qe_metricx.evaluate(
            hypotheses=translations, sources=sources, references=[]
        )
        return round(qe_metricx_result["system_score"], 4), qe_metricx_result["segments_scores"]

    
    def evaluate_sacrebleu(self):
        """Evaluates the outputs using sacrebleu."""
        
        hypotheses = [item.output for item in self.data]
        references = [item.tgt_ref for item in self.data]

        tgt_language = self.data[0].tgt_lang

        if tgt_language == 'zh': # if chinese, we use chinese tokenizer
            corpus_score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='zh').score

            # Calculate segment-level BLEU scores
            segment_scores = [
                sacrebleu.corpus_bleu( [hyp], [[ref]], tokenize='zh' ).score
                for hyp, ref in zip(hypotheses, references)
            ]

        else:
            # Calculate corpus-level BLEU score
            corpus_score = sacrebleu.corpus_bleu(hypotheses, [references]).score

            # Calculate segment-level BLEU scores
            segment_scores = [
                sacrebleu.corpus_bleu( [hyp], [[ref]] ).score
                for hyp, ref in zip(hypotheses, references)
            ]

        return round(corpus_score, 4), segment_scores

    def evaluate_chrf(self):
        """Evaluates the outputs using sacrebleu's chrF."""
        
        hypotheses = [item.output for item in self.data]
        references = [item.tgt_ref for item in self.data]

        # Calculate corpus-level chrF score. The .score attribute is used to get the float value.
        corpus_score = sacrebleu.corpus_chrf(hypotheses, [references]).score

        # Calculate segment-level chrF scores
        segment_scores = [
            sacrebleu.corpus_chrf( [hyp], [[ref]] ).score
            for hyp, ref in zip(hypotheses, references)
        ]

        return round(corpus_score, 4), segment_scores

    def evaluate_off_target_translations(self):
        """
        Evaluates off-target translations using GlotLID.
        An output is considered off-target if the predicted language
        does not match the expected target language.

        Returns:
            A tuple containing the system-level off-target rate (in %) and
            a list of segment-level scores (1 for off-target, 0 for on-target).
        """

        glotlid_model = fasttext.load_model(GlotLID_PATH)

        translations = [item.output.replace('\n', '') for item in self.data]
        target_langs = [item.tgt_lang for item in self.data]

        if not translations:
            return 0.0, []

        predictions = [glotlid_model.predict(tr) for tr in translations]
        predicted_langs = [pred[0][0] for pred in predictions]

        off_target_count = 0
        segment_scores = []
        for i in range(len(translations)):
            # Check if the predicted language matches the expected target language
            is_off_target = 1 if MAPPING_TO_FASTTEXT_LABEL[target_langs[i]] not in predicted_langs[i] else 0
            segment_scores.append( (is_off_target, predicted_langs[i].replace('__label__', '') ) )
            if is_off_target:
                off_target_count += 1
        
        # Calculate the overall off-target rate as a percentage
        total_samples = len(translations)
        system_score = (off_target_count / total_samples) * 100 if total_samples > 0 else 0.0

        return round(system_score,4), segment_scores

    def evaluate_blaser(self):
        # TO DO
        pass

    def run_evaluations(self, metrics_to_compute: Dict[str, bool]) -> List[Dict[str, Any]]:
        """
        Runs selected evaluation metrics based on the input dictionary and aggregates
        the results into a list of dictionaries, one for each sample.
        """
        # Initialize a list of dictionaries with base information for each sample.
        # This will be populated with scores as they are computed.
        results_per_sample = [
            {
                "dataset_id": item.dataset_id,
                "sample_id": item.sample_id,
                "src_lang": item.src_lang,
                "tgt_lang": item.tgt_lang,
                "output": item.output,
                "metrics": {}
            }
            for item in self.data
        ]

        # Mapping from metric key to its evaluation function and a user-friendly name
        metric_mapping = {
            'bleu': (self.evaluate_sacrebleu, "SacreBLEU"),
            'chrf': (self.evaluate_chrf, "chrF"),
            'comet': (self.evaluate_comet, "COMET"),
            'comet_kiwi': (self.evaluate_comet_kiwi, "COMET-Kiwi"),
            'xcomet': (self.evaluate_xcomet, "XCOMET"),
            'xcomet_qe': (self.evaluate_xcomet_qe, "XCOMET-QE"),
            'metricx': (self.evaluate_ref_metricx, "RefMetricX_24"),
            'metricx_qe': (self.evaluate_qe_metricx, "QEMetricX_24"),
            'glotlid': (self.evaluate_off_target_translations, "GlotLID")
        }
        
        system_scores = {}
        for metric_key, should_compute in metrics_to_compute.items():
            if should_compute:
                if metric_key in metric_mapping:
                    eval_function, metric_name = metric_mapping[metric_key]
                    print(f"Running {metric_name} evaluation...")
                    try:
                        system_score, segment_scores = eval_function()
                        system_scores[metric_name] = system_score
                        # Add the segment-level score for the current metric
                        # to each sample's dictionary in the results list.
                        for i, score in enumerate(segment_scores):
                            results_per_sample[i]["metrics"][f'{metric_key}_score'] = score
                        
                        print(f"-> {metric_name} system score: {system_score:.4f}")
                    except Exception as e:
                        print(f"Error during {metric_name} evaluation: {e}")
                else:
                    print(f"Warning: Metric '{metric_key}' is requested but no evaluation function is mapped.")

        return results_per_sample, system_scores