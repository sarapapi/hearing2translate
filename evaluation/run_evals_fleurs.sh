export HF_HOME=""
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DATASETS_CACHE="$HF_HOME/datasets"
export H2T_DATADIR=""

NAME_SYSTEM='qwen2audio-7b'
DIRECTION_PAIR='en_es'
OUTPUT_JSONL='/outputs/qwen2audio-7b/fleurs/en-es.jsonl'
MANIFEST='/manifests/fleurs/en-es.jsonl'
SAVING_FOLDER="./output_evals/fleurs/${NAME_SYSTEM}/${DIRECTION_PAIR}"
EVAL_MODE="ref_free_and_ref_based"

mkdir -p "$SAVING_FOLDER"

echo "--- [START] Processing system: ${NAME_SYSTEM} ---"

python run_evals.py --manifest-path $MANIFEST \
                    --output-path $OUTPUT_JSONL \
                    --model-name $NAME_SYSTEM \
                    --eval-type $EVAL_MODE \
                    --results-file "${SAVING_FOLDER}/results.jsonl" \
                    --summary-file "${SAVING_FOLDER}/results_summary.jsonl"