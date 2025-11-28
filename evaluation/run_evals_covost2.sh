export HUGGINGFACE_TOKEN=''
export HF_HOME="/home/aziz/hearing2translate/.cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DATASETS_CACHE="$HF_HOME/datasets"

export METRICX_CK_NAME="${HF_HOME}/hub/models--google--metricx-24-hybrid-xxl-v2p6-bfloat16/snapshots/3d9c6f7fd11b58d3eceeb68ee39fe7d2c761ee20"
export METRICX_TOKENIZER="${HF_HOME}/hub/models--google--mt5-xxl/snapshots/e07c395916dfbc315d4e5e48b4a54a1e8821b5c0"
export XCOMET_CK_NAME="${HF_HOME}/hub/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt"
export GlotLID_PATH=''

readonly BASE_PATH="/home/aziz/hearing2translate/"
readonly SAVING_BASE_DIR="./output_evals/covost2"

# ref_free_only for language pairs with NO references!!
# --- Configuration ---
# Define systems and pairs. The pairs use the primary format (with a hyphen)
SYSTEMS=('seamlessm4t' 'whisper' 'canary-v2')
DIRECTION_PAIRS=('ru-en')

# Define constant base paths.
EVAL_MODE="ref_free_only"

# --- Main Loops ---
for system in "${SYSTEMS[@]}"; do
    echo "--- [START] Processing System: ${system} ---"

    for pair in "${DIRECTION_PAIRS[@]}"; do
        # Create a version of the pair string with underscores for the saving folder.
        # This uses bash's built-in string replacement: ${variable//find/replace}.
        pair_for_saving="${pair//-/_}"
        SAVING_FOLDER="${SAVING_BASE_DIR}/${system}/${pair_for_saving}"
        RESULTS_FILE="${SAVING_FOLDER}/results.jsonl"

        # --- Check if results already exist ---
        if [ -f "$RESULTS_FILE" ]; then
            echo "--- [SKIP] Results for ${system} / ${pair} already exist. ---"
            continue # Skip to the next pair in the loop
        fi

        echo "--- [INFO] Processing Pair: ${pair} for System: ${system} ---"

        # Construct paths dynamically
        MANIFEST="${BASE_PATH}/manifests/covost2/${pair}.jsonl"
        OUTPUT_JSONL="${BASE_PATH}/outputs/${system}/covost2/${pair}.jsonl"

        # Create the target directory.
        mkdir -p "$SAVING_FOLDER"

        # Run the Python script.
        python run_evals.py \
            --manifest-path "$MANIFEST" \
            --output-path "$OUTPUT_JSONL" \
            --model-name "$system" \
            --eval-type "$EVAL_MODE" \
            --results-file "${SAVING_FOLDER}/results.jsonl" \
            --summary-file "${SAVING_FOLDER}/results_summary.jsonl"

        echo "--- [DONE] Finished Pair: ${pair} for System: ${system} ---"
    done
    echo "--- [END] Finished Processing System: ${system} ---"
done

echo "--- All systems and pairs processed successfully. ---"