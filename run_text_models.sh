#!/bin/bash

# Text model inference script for Hearing2Translate
# This script prints commands that run text model inference on selected datasets.
# It's safe for multiprocessing. Every infer.py process is locked, the lock is checked
# by others. Successful infer process is marked ok with `touch $out.ok`, so it can be
# easily inspected.

# Inspect by bare eyes what processes need to be run:
# ./run_text_models.sh <benchmark>

# Run the processes:
# ./run_text_models.sh <benchmark> | bash -v

# Multiprocessing, e.g.:
# ./run_text_models.sh <benchmark> | bash -v & ./run_text_models.sh <benchmark> | bash -v

# Check if benchmark argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <benchmark>"
    echo "Example: $0 covost2"
    exit 1
fi

BENCHMARK=$1

# Define sources and models
SOURCES=('canary-v2' 'seamlessm4t' 'whisper')
MODELS=('CohereLabs/aya-expanse-32b' 'google/gemma-3-12b-it' 'Unbabel/Tower-Plus-9B')
MODEL_NAMES=('aya' 'gemma' 'tower')

cmd() {
    echo "if mkdir $out.lock ; then python infer.py --model '$model' --in-modality text --in-file '$in_file' --transcript-file '$transcript_file' --out-file '$out' 2>&1 | tee $out.err && touch $out.ok; rm -rf $out.lock ; fi"
}

# Loop through each model
for model in "${MODELS[@]}"; do
    # Find corresponding model name for output directory
    for i in "${!MODELS[@]}"; do
        if [ "${MODELS[$i]}" = "$model" ]; then
            model_name="${MODEL_NAMES[$i]}"
            break
        fi
    done
    
    # Loop through each source
    for source in "${SOURCES[@]}"; do
        # Create output directory if it doesn't exist
        output_dir="${model_name}/${source}"
        mkdir -p "$output_dir"
        
        # Process all JSONL files in manifests/{benchmark}/
        for in_file in manifests/$BENCHMARK/*.jsonl ; do
            [ ! -f "$in_file" ] && continue  # Skip if no files match
            
            filename=$(basename "$in_file" .jsonl)
            
            # Construct transcript file path (same filename in output/{source}_asr/{benchmark}/)
            transcript_file="output/${source}_asr/${BENCHMARK}/${filename}.jsonl"
            
            # Skip if transcript file doesn't exist
            [ ! -f "$transcript_file" ] && continue
            
            # Construct output file path
            out="${output_dir}/${filename}.jsonl"
            
            # filters out successful (ok) or running (locked)
            if [ ! -f $out.ok ] && [ ! -d $out.lock ]; then
                cmd
            fi
        done
    done
done
