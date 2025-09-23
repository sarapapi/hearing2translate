#!/bin/bash

# Text model inference script for Hearing2Translate
# This script runs text model inference on selected datasets and logs results.
# It uses locking to be safe for multiprocessing. Every infer.py process is locked, 
# the lock is checked by others. Successful infer process is marked ok with `touch $out.ok`.

# Usage:
# ./run_text_models.sh <benchmark>

# Check if benchmark argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <benchmark> [model_name]"
    echo "Example: $0 covost2"
    echo "Example: $0 fleurs tower"
    echo "Available models: aya, gemma, tower"
    exit 1
fi

BENCHMARK=$1
SELECTED_MODEL=""

# Check if a specific model is requested
if [ $# -eq 2 ]; then
    SELECTED_MODEL="$2"
    # Validate the model name
    valid_models=("aya" "gemma" "tower")
    if [[ ! " ${valid_models[@]} " =~ " ${SELECTED_MODEL} " ]]; then
        echo "Error: Invalid model name '$SELECTED_MODEL'"
        echo "Available models: ${valid_models[@]}"
        exit 1
    fi
    echo "Running only model: $SELECTED_MODEL"
fi

LOG_FILE="run_text_models_${BENCHMARK}_$(date +%Y%m%d_%H%M%S).log"
if [ -n "$SELECTED_MODEL" ]; then
    LOG_FILE="run_text_models_${BENCHMARK}_${SELECTED_MODEL}_$(date +%Y%m%d_%H%M%S).log"
fi

# Initialize log file
echo "==================================================" | tee "$LOG_FILE"
echo "Text Model Inference Run - $(date)" | tee -a "$LOG_FILE"
echo "Benchmark: $BENCHMARK" | tee -a "$LOG_FILE"
if [ -n "$SELECTED_MODEL" ]; then
    echo "Selected Model: $SELECTED_MODEL" | tee -a "$LOG_FILE"
fi
echo "==================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Define sources and models
SOURCES=('canary-v2' 'seamlessm4t' 'whisper')
MODELS=('CohereLabs/aya-expanse-32b' 'google/gemma-3-12b-it' 'Unbabel/Tower-Plus-9B')
MODEL_NAMES=('aya' 'gemma' 'tower')

# Counters for statistics
total_tasks=0
successful_tasks=0
failed_tasks=0
skipped_tasks=0

run_inference() {
    local model="$1"
    local in_file="$2"
    local transcript_file="$3"
    local out="$4"
    
    echo "[$(date '+%H:%M:%S')] Starting: $model -> $(basename $out)" | tee -a "$LOG_FILE"
    
    # Check if we can acquire lock
    if mkdir "$out.lock" 2>/dev/null; then
        # Run the inference
        start_time=$(date +%s)
        if python infer.py --model "$model" --in-modality text --in-file "$in_file" --transcript-file "$transcript_file" --out-file "$out" 2>&1 | tee "$out.err"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            touch "$out.ok"
            echo "[$(date '+%H:%M:%S')] SUCCESS: $model -> $(basename $out) (${duration}s)" | tee -a "$LOG_FILE"
            ((successful_tasks++))
        else
            exit_code=$?
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "[$(date '+%H:%M:%S')] FAILED: $model -> $(basename $out) (${duration}s)" | tee -a "$LOG_FILE"
            echo "  Error details saved in: $out.err" | tee -a "$LOG_FILE"
            ((failed_tasks++))
        fi
        # Clean up lock
        rm -rf "$out.lock"
    else
        echo "[$(date '+%H:%M:%S')] LOCKED: $model -> $(basename $out) (another process running)" | tee -a "$LOG_FILE"
        ((skipped_tasks++))
    fi
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
    
    # Skip if a specific model is selected and this isn't it
    if [ -n "$SELECTED_MODEL" ] && [ "$model_name" != "$SELECTED_MODEL" ]; then
        continue
    fi
    
    echo "Processing model: $model" | tee -a "$LOG_FILE"
    
    # Loop through each source
    for source in "${SOURCES[@]}"; do
        echo "  Processing source: $source" | tee -a "$LOG_FILE"
        
        # Create output directory if it doesn't exist
        output_dir="outputs/${model_name}_${source}/${BENCHMARK}"
        mkdir -p "$output_dir"
        
        # Process all JSONL files in manifests/{benchmark}/
        for in_file in manifests/$BENCHMARK/*.jsonl ; do
            [ ! -f "$in_file" ] && continue  # Skip if no files match
            
            filename=$(basename "$in_file" .jsonl)
            
            # Construct transcript file path (same filename in outputs/{source}_asr/{benchmark}/)
            transcript_file="outputs/${source}_asr/${BENCHMARK}/${filename}.jsonl"
            
            # Skip if transcript file doesn't exist
            if [ ! -f "$transcript_file" ]; then
                echo "    SKIP: $filename (no transcript: $transcript_file)" | tee -a "$LOG_FILE"
                continue
            fi
            
            # Construct output file path
            out="${output_dir}/${filename}.jsonl"
            
            # Check if already successful or currently running
            if [ -f "$out.ok" ]; then
                echo "    SKIP: $filename (already completed)" | tee -a "$LOG_FILE"
                ((skipped_tasks++))
            elif [ -d "$out.lock" ]; then
                echo "    SKIP: $filename (currently running)" | tee -a "$LOG_FILE"
                ((skipped_tasks++))
            else
                ((total_tasks++))
                run_inference "$model" "$in_file" "$transcript_file" "$out"
            fi
        done
    done
    echo "" | tee -a "$LOG_FILE"
done

# Print final statistics
echo "==================================================" | tee -a "$LOG_FILE"
echo "FINAL STATISTICS" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Total tasks attempted: $total_tasks" | tee -a "$LOG_FILE"
echo "Successful: $successful_tasks" | tee -a "$LOG_FILE"
echo "Failed: $failed_tasks" | tee -a "$LOG_FILE"
echo "Skipped: $skipped_tasks" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $failed_tasks -gt 0 ]; then
    echo "FAILED TASKS - Check these error files:" | tee -a "$LOG_FILE"
    find outputs/ -name "*.err" -newer "$LOG_FILE" -exec echo "  {}" \; | tee -a "$LOG_FILE"
fi

echo "Log saved to: $LOG_FILE"

