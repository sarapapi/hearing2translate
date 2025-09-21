import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from util import Evaluator

# --- Configuration ---
# Use a constant for evaluation types for clarity and easy modification.
EVALUATION_MODES: Dict[str, Dict[str, bool]] = {
    'ref_free_only': {
        'bleu': False, 'chrf': False, 'comet': False, 
        'comet_kiwi': False, 'xcomet': False, 'xcomet_qe': True, 
        'metricx': False, 'metricx_qe': True, 'glotlid': False, 'linguapy': True
    },
    'ref_free_and_ref_based': {
        'bleu': True, 'chrf': True, 'comet': False, 
        'comet_kiwi': False, 'xcomet': True, 'xcomet_qe': True, 
        'metricx': True, 'metricx_qe': True, 'glotlid': False, 'linguapy': True
    },
}

def save_to_jsonl(data, file_path):
    """
    Saves data to a specified file in JSONL format.

    Args:
        data: A list of dictionaries or a single dictionary to save.
        file_path: The Path object for the output file.
    """
    try:
        with file_path.open('w', encoding='utf-8') as f:
            if isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else: # It's a single dictionary for the summary
                f.write(json.dumps(data) + '\n')
        logging.info(f"Successfully saved results to {file_path}")
    except IOError as e:
        logging.error(f"Failed to write to file {file_path}: {e}")
    except TypeError as e:
        logging.error(f"JSON serialization failed for {file_path}: {e}")


def main():
    """
    Main function for model evaluation process.
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup robust argument parsing
    parser = argparse.ArgumentParser(description='Run machine translation evaluations.')
    parser.add_argument('--manifest-path', type=Path, required=True, help='Path to the manifest file (input data).')
    parser.add_argument('--output-path', type=Path, required=True, help='Path to the file with model predictions.')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model being evaluated.')
    parser.add_argument(
        '--eval-type', 
        type=str, 
        required=True,
        choices=EVALUATION_MODES.keys(),
        help='Type of evaluation to run.'
    )
    parser.add_argument('--results-file', type=Path, required=True, help='Output file for detailed results (JSONL).')
    parser.add_argument('--summary-file', type=Path, required=True, help='Output file for summary results (JSONL).')
    
    args = parser.parse_args()

    # Select the metrics based on the chosen evaluation type
    metrics_to_compute = EVALUATION_MODES[args.eval_type]
    logging.info(f"Starting evaluation for model: '{args.model_name}' with mode: '{args.eval_type}'")

    # Initialize the evaluator
    evaluator = Evaluator(args.manifest_path, args.output_path, args.model_name)

    # Run evaluations
    logging.info("Running evaluations... This may take a while.")
    results_list, summary_dict = evaluator.run_evaluations(metrics_to_compute)
    logging.info("Evaluations completed.")

    # Save the detailed and summary results using the helper function
    save_to_jsonl(results_list, args.results_file)
    save_to_jsonl(summary_dict, args.summary_file)


if __name__ == "__main__":
    main()