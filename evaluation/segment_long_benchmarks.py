import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mweralign.mweralign import SPSegmenter, align_texts

# --- Data Structures ---

@dataclass
class MergedData:
    """
    Schema for the combined data, ensuring all fields are defined upfront.
    """
    dataset_id: str
    sample_id: int
    src_lang: str
    tgt_lang: str
    output: str
    doc_id: Optional[str]
    references_segmented: Dict[str, str] = field(default_factory=dict)
    src_ref: Optional[str] = None
    tgt_ref: Optional[Dict[str, Any]] = None
    src_audio: Optional[str] = None
    benchmark_metadata: Optional[Dict[str, Any]] = None

# --- Helper Functions ---

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries with robust error handling."""
    if not file_path.is_file():
        logging.error(f"Error: File not found at {file_path}")
        return []
    try:
        with file_path.open('r', encoding='utf-8') as f:
            # Use a list comprehension for a more concise implementation.
            return [json.loads(line) for line in f]
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON in {file_path}")
        return []
    except IOError as e:
        logging.error(f"An I/O error occurred while reading {file_path}: {e}")
        return []

def _tokenize_line(line: str, segmenter: SPSegmenter) -> str:
    """Tokenizes a single line, handling special '###' or tab separators."""
    separator = " ### " if " ### " in line else "\t" if "\t" in line else None

    if separator:
        pieces = line.strip().split(separator)
        # The underlying C++ binary uses '###', so we standardize to that.
        return " ### ".join(" ".join(segmenter.encode(p)) for p in pieces)
    else:
        return " ".join(segmenter.encode(line.strip()))

def tokenize_text(text_lines: List[str], segmenter: SPSegmenter) -> str:
    """Tokenizes a list of strings and joins them into a single block for alignment."""
    return "\n".join(_tokenize_line(line, segmenter) for line in text_lines)

def get_alignment(
    refs: List[str],
    hyp: str,
    language: str,
    segmenter: SPSegmenter,
    no_detok: bool = False
) -> Optional[Dict[str, str]]:
    """
    Performs mWER alignment for a single hypothesis against multiple references.
    """
    # Tokenize inputs using the helper function.
    hyp_str = tokenize_text([hyp], segmenter)
    ref_str = tokenize_text(refs, segmenter)

    logging.info(f"Aligning {len(hyp_str.split())} hypothesis tokens to {len(refs)} references.")

    # This param adjusts the AS-WER algorithm for pre-tokenized text.
    is_tokenized = language not in ["ja", "zh"]

    try:
        alignment_result = align_texts(ref_str, hyp_str, is_tokenized=is_tokenized)

        # Process and detokenize results.
        aligned_segments = {}
        # Zip original references with aligned output lines to create the final mapping.
        for original_ref, aligned_line in zip(refs, alignment_result.split("\n")):
            if not no_detok:
                aligned_line = segmenter.decode(aligned_line)
            aligned_segments[original_ref] = aligned_line
        return aligned_segments

    except Exception as e:
        # Catching a broad exception here, but logging the specific error.
        logging.error(f"An error occurred during alignment: {e}")
        return None

def main():
    """Main function to run the mWER alignment process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Align a long model output with short reference segments using mWER-align.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--manifest-long-path', type=Path, required=True,
                        help='Path to the JSONL manifest for long-form audio/text.')
    parser.add_argument('--manifest-short-path', type=Path, required=True,
                        help='Path to the JSONL manifest with short reference segments.')
    parser.add_argument('--model-output-path', type=Path, required=True,
                        help='Path to the JSONL file with model predictions.')
    parser.add_argument('--output-segmented-file', type=Path, required=True,
                        help='Output file path for segmented model outputs (JSONL).')
    parser.add_argument('--tokenizer-path', type=Path, required=True,
                        help='Path to the SentencePiece model for tokenization.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    logging.info("Loading data files...")
    long_manifest = load_jsonl(args.manifest_long_path)
    short_manifest = load_jsonl(args.manifest_short_path)
    model_output = load_jsonl(args.model_output_path)

    if not all((long_manifest, short_manifest, model_output)):
        logging.fatal("One or more input files are empty or could not be loaded. Exiting.")
        return

    # --- 2. Pre-process and Merge Data ---
    logging.info("Preprocessing and merging data...")
    # OPTIMIZATION: Create an efficient lookup for short references by document ID.
    # This avoids a very slow O(N*M) loop.
    short_manifest_by_doc = defaultdict(list)
    for item in short_manifest:
        short_manifest_by_doc[item['doc_id']].append(item)

    # Use a dictionary for faster, more robust lookup of model outputs by a unique ID.
    output_by_id = {item['sample_id']: item for item in model_output}
    
    merged_data: List[MergedData] = []
    for input_item in long_manifest:
        sample_id = input_item['sample_id']
        output_item = output_by_id.get(sample_id)
        if not output_item:
            logging.warning(f"No model output found for sample_id: {sample_id}. Skipping.")
            continue
        
        doc_id = input_item.get('doc_id')
        short_segments = short_manifest_by_doc.get(doc_id, [])
        references_segmented = {
            item.get('src_ref', ''): item.get('tgt_ref', '') for item in short_segments
        }

        merged_data.append(
            MergedData(
                dataset_id=output_item['dataset_id'],
                sample_id=output_item['sample_id'],
                src_lang=output_item['src_lang'],
                tgt_lang=output_item['tgt_lang'],
                output=output_item['output'],
                doc_id=doc_id,
                references_segmented=references_segmented,
                **{k: input_item.get(k) for k in ['src_ref', 'tgt_ref', 'src_audio', 'benchmark_metadata']}
            )
        )

    # --- 3. Perform Alignment ---
    logging.info(f"Initializing tokenizer from {args.tokenizer_path}...")
    # Initialize the segmenter only ONCE.
    segmenter = SPSegmenter(str(args.tokenizer_path))

    logging.info("Performing mWER alignment for all documents...")
    all_alignments: Dict[str, str] = {}
    for item in merged_data:
        if not item.references_segmented:
            logging.warning(f"No references found for doc_id: {item.doc_id}. Skipping alignment.")
            continue

        refs = list(item.references_segmented.values())
        alignments = get_alignment(refs, item.output, item.tgt_lang, segmenter)
        if alignments:
            all_alignments.update(alignments)

    # --- 4. Generate and Write Final Output ---
    logging.info("Generating final segmented output file...")
    final_output_data = []
    for item in short_manifest:
        aligned_output = all_alignments.get(item.get('tgt_ref'))
        if aligned_output is None:
            logging.warning(f"Could not find alignment for reference in sample_id: {item['sample_id']}.")
            continue
        
        final_output_data.append({
            "dataset_id": item['dataset_id'],
            "sample_id": item['sample_id'],
            "src_lang": item['src_lang'],
            "tgt_lang": item['tgt_lang'],
            "output": aligned_output
        })

    try:
        with args.output_segmented_file.open("w", encoding="utf-8") as f:
            for entry in final_output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"Successfully wrote {len(final_output_data)} segmented outputs to {args.output_segmented_file}")
    except IOError as e:
        logging.fatal(f"Could not write to output file {args.output_segmented_file}: {e}")

if __name__ == "__main__":
    main()