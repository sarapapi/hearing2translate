import numpy as np
from metrics.bleurt.metric import BLEURT
from metrics.comet.metric import BaseCOMET
from metrics.comet_kiwi.metric import COMETKiwi
from metrics.xcomet.metric import XCOMET, XCOMET_QE
from metrics.metricx.metric import RefMetricX, QEMetricX
import sacrebleu
import random
import json

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

BLEURT_CK_NAME=''
COMET_CK_NAME=''
COMET_KIWI_CK_NAME='Unbabel/wmt22-cometkiwi-da'
XCOMET_CK_NAME=''
METRICX_CK_NAME=''

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

    def evaluate_bleurt(self):
        """Evaluates the outputs using BLEURT."""
        self.bleurt = BLEURT(BLEURT_CK_NAME)
        batch_size = 8

        targets = [[item.tgt_ref] for item in self.data]
        translations = [item.output for item in self.data]

        bleurt_result = self.bleurt.evaluate(translations, targets, batch_size)
        return bleurt_result["system_score"], bleurt_result['segments_scores']

    def evaluate_comet(self):
        """Evaluates the outputs using BaseCOMET."""
        self.comet = BaseCOMET(COMET_CK_NAME)
        batch_size = 8

        sources = [[item.src_ref] for item in self.data]
        targets = [[item.tgt_ref] for item in self.data]
        translations = [item.output for item in self.data]

        comet_result = self.comet.evaluate(translations, targets, sources, batch_size )
        return comet_result["system_score"], comet_result["segments_scores"]

    def evaluate_comet_kiwi(self):
        """Evaluates the outputs using COMETKiwi (QE metric)."""
        self.comet_kiwi = COMETKiwi(COMET_KIWI_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        comet_kiwi_result = self.comet_kiwi.evaluate(translations, sources, batch_size)
        return comet_kiwi_result["system_score"], comet_kiwi_result["segments_scores"]


    def evaluate_xcomet(self):
        """Evaluates the outputs using XCOMET."""
        self.xcomet = XCOMET(XCOMET_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        targets = [[item.tgt_ref] for item in self.data]
        translations = [item.output for item in self.data]

        xcomet_result = self.xcomet.evaluate(translations, targets, sources, batch_size)
        return xcomet_result["system_score"], xcomet_result["segments_scores"]


    def evaluate_xcomet_qe(self):
        """Evaluates the outputs using XCOMET_QE (QE metric)."""
        self.xcomet_qe = XCOMET_QE(XCOMET_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        xcomet_qe_result = self.xcomet_qe.evaluate(translations, [], sources, batch_size)
        return xcomet_qe_result["system_score"], xcomet_qe_result["segments_scores"]


    def evaluate_ref_metricx(self):
        """Evaluates the outputs using RefMetricX."""
        self.ref_metricx = RefMetricX(METRICX_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        targets = [[item.tgt_ref] for item in self.data]
        translations = [item.output for item in self.data]

        ref_metricx_result = self.ref_metricx.evaluate(
            translations=translations, references=targets, sources=sources, batch_size=batch_size
        )
        return ref_metricx_result["system_score"], ref_metricx_result["segments_scores"]


    def evaluate_qe_metricx(self):
        """Evaluates the outputs using QEMetricX (QE metric)."""
        self.qe_metricx = QEMetricX(METRICX_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        qe_metricx_result = self.qe_metricx.evaluate(
            translations=translations, sources=sources, batch_size=batch_size
        )
        return qe_metricx_result["system_score"], qe_metricx_result["segments_scores"]

    
    def evaluate_sacrebleu(self):
        """Evaluates the outputs using sacrebleu."""
        
        hypotheses = [item.output for item in self.data]
        references = [[item.tgt_ref] for item in self.data]

        tgt_language = self.data[0]['tgt_lang']

        if tgt_language == 'zh': # if chinese, we use chinese tokenizer
            corpus_score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='zh').score

            # Calculate segment-level BLEU scores
            segment_scores = [
                sacrebleu.sentence_bleu(hyp, ref, tokenize='zh').score
                for hyp, ref in zip(hypotheses, references)
            ]

        else:
            # Calculate corpus-level BLEU score
            corpus_score = sacrebleu.corpus_bleu(hypotheses, [references]).score

            # Calculate segment-level BLEU scores
            segment_scores = [
                sacrebleu.sentence_bleu(hyp, ref).score
                for hyp, ref in zip(hypotheses, references)
            ]

        return corpus_score, segment_scores

    def evaluate_off_target_translations(self):
        # TO DO
        pass

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
            'bleurt': (self.evaluate_bleurt, "BLEURT"),
            'comet': (self.evaluate_comet, "COMET"),
            'comet_kiwi': (self.evaluate_comet_kiwi, "COMET-Kiwi"),
            'xcomet': (self.evaluate_xcomet, "XCOMET"),
            'xcomet_qe': (self.evaluate_xcomet_qe, "XCOMET-QE"),
            'metricx': (self.evaluate_ref_metricx, "RefMetricX"),
            'metricx_qe': (self.evaluate_qe_metricx, "QEMetricX"),
        }

        for metric_key, should_compute in metrics_to_compute.items():
            if should_compute:
                if metric_key in metric_mapping:
                    eval_function, metric_name = metric_mapping[metric_key]
                    print(f"Running {metric_name} evaluation...")
                    try:
                        system_score, segment_scores = eval_function()

                        # Add the segment-level score for the current metric
                        # to each sample's dictionary in the results list.
                        for i, score in enumerate(segment_scores):
                            results_per_sample[i]["metrics"][f'{metric_key}_score'] = score
                        
                        print(f"-> {metric_name} system score: {system_score:.4f}")
                    except Exception as e:
                        print(f"Error during {metric_name} evaluation: {e}")
                else:
                    print(f"Warning: Metric '{metric_key}' is requested but no evaluation function is mapped.")

        return results_per_sample