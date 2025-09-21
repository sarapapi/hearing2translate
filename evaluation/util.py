import numpy as np
from metrics.comet.metric import BaseCOMET
from metrics.comet_kiwi.metric import COMETKiwi
from metrics.xcomet.metric import XCOMET, XCOMET_QE
from metrics.metricx.metric import RefMetricX_24, QEMetricX_24
import sacrebleu
import json
import fasttext
import os
from lingua import LanguageDetectorBuilder
import torch
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# note that downloading Comet models requires HF token and gated access
COMET_CK_NAME=''
COMET_KIWI_CK_NAME=''

XCOMET_CK_NAME=os.environ.get("XCOMET_CK_NAME")
METRICX_CK_NAME = os.environ.get("METRICX_CK_NAME")
METRICX_TOKENIZER = os.environ.get("METRICX_TOKENIZER")
GlotLID_PATH = os.environ.get("GLOTLID_PATH")

MAPPING_TO_FASTTEXT_LABEL = {
    'it': 'ita_Latn', 'es':'spa_Latn', 'de':'deu_Latn', 'zh':'cmn', 
    'pt': 'por_Latn', 'en':'eng_Latn', 'fr':'fra_Latn', 'nl': 'nld_Latn'
}

MAPPING_TO_LINGUA_LABEL = {
    'it': 'ITALIAN', 'es':'SPANISH', 'de':'GERMAN', 'zh':'CHINESE', 
    'pt': 'PORTUGUESE', 'en':'ENGLISH', 'fr':'FRENCH', 'nl': 'DUTCH'
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
            logging.info(f"Error: File not found at {file_path}")
            return []
        except json.JSONDecodeError:
            logging.info(f"Error: Could not decode JSON in {file_path}")
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
            logging.info(
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
                logging.info(f"Warning: Missing key {e} in data at line index {i}. Skipping this entry.")

        self.is_multi_ref = isinstance(merged_data[0].tgt_ref, dict) if merged_data and merged_data[0].tgt_ref else False
        if self.is_multi_ref:
            logging.info('Multi-reference dataset detected!')

        return merged_data

    def get_all_data(self) -> List[MergedData]:
        """Returns all merged data points as a list."""
        return self.data

    # --------------------------------------------------------------------------
    # Metric Evaluation Functions
    # --------------------------------------------------------------------------

    def evaluate_comet(self):
        """Evaluates the outputs using BaseCOMET."""
        comet = BaseCOMET(COMET_CK_NAME)
        batch_size = 8

        sources = [ item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        if not self.is_multi_ref:
            targets = [ item.tgt_ref for item in self.data]
            comet_result = comet.evaluate(translations, targets, sources, batch_size )
            return round(comet_result["system_score"], 4), comet_result["segments_scores"]

        else:
            ref_keys = sorted(self.data[0].tgt_ref.keys())
            refs_by_key = {key: [str(item.tgt_ref.get(key, '')) for item in self.data] for key in ref_keys}
            
            system_scores = []
            segment_scores_list = []

            for ref_key, targets in refs_by_key.items():
                logging.info(f"Computing COMET for reference: {ref_key}")
                result = comet.evaluate(translations, targets, sources, batch_size)
                system_scores.append(result["system_score"])
                segment_scores_list.append(result["segments_scores"])

            avg_system_score = np.mean(system_scores)
            avg_segment_scores = np.mean(np.array(segment_scores_list), axis=0).tolist()

            return round(avg_system_score, 4), avg_segment_scores

    def evaluate_comet_kiwi(self):
        """Evaluates the outputs using COMETKiwi (QE metric)."""
        comet_kiwi = COMETKiwi(COMET_KIWI_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        comet_kiwi_result = comet_kiwi.evaluate(translations, sources, batch_size)
        return round(comet_kiwi_result["system_score"], 4), comet_kiwi_result["segments_scores"]


    def evaluate_xcomet(self):
        """Evaluates the outputs using XCOMET."""
        xcomet = XCOMET(XCOMET_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        if not self.is_multi_ref:
            targets = [ item.tgt_ref for item in self.data]
            xcomet_result = xcomet.evaluate(translations, targets, sources, batch_size)
        else:

            ref_keys = sorted(self.data[0].tgt_ref.keys())
            refs_by_key = {key: [str(item.tgt_ref.get(key, '')) for item in self.data] for key in ref_keys}
            
            system_scores = []
            segment_scores_list = []
            for ref_key, targets in refs_by_key.items():
                logging.info(f"Computing XCOMET for reference: {ref_key}")
                result = xcomet.evaluate(translations, targets, sources, batch_size)
                system_scores.append(result["system_score"])
                segment_scores_list.append(result["segments_scores"])

            avg_system_score = np.mean(system_scores)
            avg_segment_scores = np.mean(np.array(segment_scores_list), axis=0).tolist()
            xcomet_result = {"system_score": avg_system_score, "segments_scores": avg_segment_scores}

        xcomet.model.to('cpu')
        del xcomet
        torch.cuda.empty_cache()

        return round(xcomet_result["system_score"], 4), xcomet_result["segments_scores"]


    def evaluate_xcomet_qe(self):
        """Evaluates the outputs using XCOMET_QE (QE metric)."""
        xcomet_qe = XCOMET_QE(XCOMET_CK_NAME)
        batch_size = 8

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        xcomet_qe_result = xcomet_qe.evaluate(translations, [], sources, batch_size)

        xcomet_qe.model.to('cpu')
        del xcomet_qe
        torch.cuda.empty_cache()

        return round(xcomet_qe_result["system_score"], 4), xcomet_qe_result["segments_scores"]


    def evaluate_ref_metricx(self):
        """Evaluates the outputs using RefMetricX_24."""
        ref_metricx = RefMetricX_24(METRICX_TOKENIZER, METRICX_CK_NAME)

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        if not self.is_multi_ref:
            targets = [item.tgt_ref for item in self.data]
            ref_metricx_result = ref_metricx.evaluate(
                hypotheses=translations, references=targets, sources=sources
            )

        else:
            ref_keys = sorted(self.data[0].tgt_ref.keys())
            refs_by_key = {key: [str(item.tgt_ref.get(key, '')) for item in self.data] for key in ref_keys}

            system_scores = []
            segment_scores_list = []
            for ref_key, targets in refs_by_key.items():
                logging.info(f"Computing RefMetricX_24 for reference: {ref_key}")
                result = ref_metricx.evaluate(
                    hypotheses=translations, references=targets, sources=sources
                )
                system_scores.append(result["system_score"])
                segment_scores_list.append(result["segments_scores"])

            avg_system_score = np.mean(system_scores)
            avg_segment_scores = np.mean(np.array(segment_scores_list), axis=0).tolist()
            ref_metricx_result = {"system_score": avg_system_score, "segments_scores": avg_segment_scores}  

        del ref_metricx
        torch.cuda.empty_cache()

        return round(ref_metricx_result["system_score"], 4), ref_metricx_result["segments_scores"]


    def evaluate_qe_metricx(self):
        """Evaluates the outputs using QEMetricX_24 (QE metric)."""
        qe_metricx = QEMetricX_24(METRICX_TOKENIZER, METRICX_CK_NAME)

        sources = [item.src_ref for item in self.data]
        translations = [item.output for item in self.data]

        qe_metricx_result = qe_metricx.evaluate(
            hypotheses=translations, sources=sources, references=[]
        )

        del qe_metricx
        torch.cuda.empty_cache()

        return round(qe_metricx_result["system_score"], 4), qe_metricx_result["segments_scores"]

    
    def evaluate_sacrebleu(self):
        """Evaluates the outputs using sacrebleu."""
        
        hypotheses = [item.output for item in self.data]
        tgt_language = self.data[0].tgt_lang

        corpus_refs = []
        segment_refs_grouped = []

        if not self.is_multi_ref:
            single_ref_list = [str(item.tgt_ref) for item in self.data]
            corpus_refs = [single_ref_list]
            segment_refs_grouped = [[ref] for ref in single_ref_list]
        else:
            ref_keys = sorted(self.data[0].tgt_ref.keys())
            refs_by_key = {key: [str(item.tgt_ref.get(key, '')) for item in self.data] for key in ref_keys}
            corpus_refs = list(refs_by_key.values())
            
            for i in range(len(self.data)):
                segment_refs_grouped.append([refs_by_key[key][i] for key in ref_keys])


        if tgt_language == 'zh': # if chinese, we use chinese tokenizer
            corpus_score = sacrebleu.corpus_bleu(hypotheses, corpus_refs, tokenize='zh').score

            # Calculate segment-level BLEU scores
            segment_scores = [
                sacrebleu.corpus_bleu([hyp], [refs], tokenize='zh').score
                for hyp, refs in zip(hypotheses, segment_refs_grouped)
            ]

        else:
            # Calculate corpus-level BLEU score
            corpus_score = sacrebleu.corpus_bleu(hypotheses, corpus_refs).score

            # Calculate segment-level BLEU scores
            segment_scores = [
                sacrebleu.corpus_bleu([hyp], [refs]).score
                for hyp, refs in zip(hypotheses, segment_refs_grouped)
            ]

        return round(corpus_score, 4), segment_scores

    def evaluate_chrf(self):
        """Evaluates the outputs using sacrebleu's chrF."""
        
        hypotheses = [item.output for item in self.data]

        corpus_refs = []
        segment_refs_grouped = []

        if not self.is_multi_ref:
            single_ref_list = [str(item.tgt_ref) for item in self.data]
            corpus_refs = [single_ref_list]
            segment_refs_grouped = [[ref] for ref in single_ref_list]
        else:
            ref_keys = sorted(self.data[0].tgt_ref.keys())
            refs_by_key = {key: [str(item.tgt_ref.get(key, '')) for item in self.data] for key in ref_keys}
            corpus_refs = list(refs_by_key.values())
            
            for i in range(len(self.data)):
                segment_refs_grouped.append([refs_by_key[key][i] for key in ref_keys])
        

        # Calculate corpus-level chrF score. The .score attribute is used to get the float value.
        corpus_score = sacrebleu.corpus_chrf(hypotheses, corpus_refs).score
        segment_scores = [
            sacrebleu.corpus_chrf([hyp], [refs]).score
            for hyp, refs in zip(hypotheses, segment_refs_grouped)
        ]

        return round(corpus_score, 4), segment_scores

    def evaluate_off_target_translations_glotLID(self):
        """
        Evaluates off-target translations using GlotLID.
        An output is considered off-target if the predicted language
        does not match the expected target language.

        Returns:
            A tuple containing the system-level off-target rate (in %) and
            a list of segment-level scores (1 for off-target, 0 for on-target).
        """

        glotlid_model = fasttext.load_model(GlotLID_PATH)
        logging.info('glotlid model loaded!')

        translations = [item.output.replace('\n', '') for item in self.data]
        target_langs = [item.tgt_lang for item in self.data]

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

    def evaluate_off_target_translations_linguapy(self):
        """
        Evaluates off-target translations using LinguaPY.
        An output is considered off-target if the predicted language
        does not match the expected target language.

        Returns:
            A tuple containing the system-level off-target rate (in %) and
            a list of segment-level scores (1 for off-target, 0 for on-target).
        """

        lingua_model = LanguageDetectorBuilder.from_all_spoken_languages().build()
        logging.info('Lingua model loaded!')

        translations = [item.output.replace('\n', '') for item in self.data]
        target_langs = [item.tgt_lang for item in self.data]

        predictions = [lingua_model.detect_language_of(tr) for tr in translations]

        try:
            predicted_langs = [pred.name for pred in predictions]
        except:
            predicted_langs = [getattr(pred, 'name', 'UNKNOWN') for pred in predictions]

        off_target_count = 0
        segment_scores = []
        for i in range(len(translations)):
            # Check if the predicted language matches the expected target language
            is_off_target = 1 if MAPPING_TO_LINGUA_LABEL[target_langs[i]] not in predicted_langs[i] else 0
            segment_scores.append( (is_off_target, predicted_langs[i] ) )
            if is_off_target:
                off_target_count += 1
        
        # Calculate the overall off-target rate as a percentage
        total_samples = len(translations)
        system_score = (off_target_count / total_samples) * 100 if total_samples > 0 else 0.0

        return round(system_score,4), segment_scores

    def run_evaluations(self, metrics_to_compute: Dict[str, bool]) -> List[Dict[str, Any]]:
        """
        Runs selected evaluation metrics, including strict scoring for off-target translations.
        """
        results_per_sample = [
            {"dataset_id": item.dataset_id, "sample_id": item.sample_id,
            "src_lang": item.src_lang, "tgt_lang": item.tgt_lang,
            "output": item.output, "metrics": {}}
            for item in self.data
        ]

        metric_mapping = {
            'bleu': (self.evaluate_sacrebleu, "SacreBLEU"),
            'chrf': (self.evaluate_chrf, "chrF"),
            'comet': (self.evaluate_comet, "COMET"),
            'comet_kiwi': (self.evaluate_comet_kiwi, "COMET-Kiwi"),
            'xcomet': (self.evaluate_xcomet, "XCOMET"),
            'xcomet_qe': (self.evaluate_xcomet_qe, "XCOMET-QE"),
            'metricx': (self.evaluate_ref_metricx, "RefMetricX_24"),
            'metricx_qe': (self.evaluate_qe_metricx, "QEMetricX_24"),
            'glotlid': (self.evaluate_off_target_translations_glotLID, "GlotLID"),
            'linguapy': (self.evaluate_off_target_translations_linguapy, "LinguaPy")
        }
        
        # --- Run all requested metrics ---
        system_scores = {}
        for metric_key, should_compute in metrics_to_compute.items():
            if should_compute and metric_key in metric_mapping:
                eval_function, metric_name = metric_mapping[metric_key]
                logging.info(f"Running {metric_name} evaluation...")

                system_score, segment_scores = eval_function()
                system_scores[metric_name] = system_score

                for i, score in enumerate(segment_scores):
                    results_per_sample[i]["metrics"][f'{metric_key}_score'] = score
                
                logging.info(f"-> {metric_name} system score: {system_score:.4f}")

        # --- Calculate strict scores using a language detector ---
        logging.info("Calculating strict system scores if applicable...")
        penalty_by_metric = {
            'metricx': 25, 'metricx_qe': 25, 
            'comet': 0, 'comet_kiwi': 0, 
            'xcomet': 0, 'xcomet_qe': 0
        }

        preprocess_score = lambda score, off_target, metric: score if off_target == 0 else penalty_by_metric[metric]
        strict_metrics = penalty_by_metric.keys()
        
        for detector in ['linguapy', 'glotlid']:
            if metrics_to_compute.get(detector, False):
                logging.info(f"Using {detector} for strict scoring...")
                # Loop through metrics that need a strict score
                for metric_key in strict_metrics:
                    # Check if the metric to be penalized was actually computed
                    if metrics_to_compute.get(metric_key, False):
                        _, metric_name = metric_mapping[metric_key]
                        
                        strict_scores = [
                            preprocess_score(
                                res["metrics"][f'{metric_key}_score'], 
                                res["metrics"][f'{detector}_score'][0], # The flag is the first element of the tuple
                                metric_key
                            ) for res in results_per_sample
                        ]
                        
                        mean_score = round(np.mean(strict_scores), 4)
                        system_scores[f"{metric_name}-Strict-{detector}"] = mean_score
                        logging.info(f"-> {metric_name}-Strict-{detector} system score: {mean_score:.4f}")

        return results_per_sample, system_scores