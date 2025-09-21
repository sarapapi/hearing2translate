# --- USAGE ---
# Description: This script processes translation data for a given system.
# It prepares the data, runs fast_align, and then generates predictions
# for different dataset variations (neutral, pro-stereotypical, anti-stereotypical).
#
# Example: sh eval_WinoST.sh seamlessm4t /outputs/seamlessm4t/winoST/en-it.jsonl /manifests/winoST/en-it.jsonl it awesome_align

# ----------------

# 1. VALIDATE INPUT ARGUMENTS
if [ "$#" -ne 5 ]; then
    echo "ERROR: Incorrect number of arguments."
    echo "Usage: $0 <system_name> <output_jsonl_path> <input_jsonl_path> <tgt_language> <alignment_method>"
    exit 1
fi

# 2. CONFIGURATION & SETUP
# --- System-specific variables from command-line arguments ---
NAME_SYSTEM=$1         # e.g., "google", "deepl", "chatgpt"
INPUT_JSONL=$2         # The path to the system's output .jsonl file
MANIFEST=$3
TARGET_LANG=$4
ALIGNMENT_METHOD=$5

# --- Static configuration ---
DIRECTION_PAIR="en_${TARGET_LANG}"

# --- Define dataset paths ---
# We store the dataset names in an array to loop over them later
DATASET_NAMES=("en" "en_pro" "en_anti")
BASE_DATA_PATH="./metrics/winoMT/data"

# --- Output directories ---
DIR_SAVE="./metrics/winoMT/translations/${NAME_SYSTEM}"
OUT_FOLDER="./output_evals/winoST/${NAME_SYSTEM}"

# --- Create directories if they don't exist ---
mkdir -p "$DIR_SAVE"
mkdir -p "$OUT_FOLDER"

echo "--- [START] Processing system: ${NAME_SYSTEM} ---"

# 3. PREPARE DATA & ALIGN
# --- Prepare WinoST data from the system's output ---
# This step creates the bilingual text file needed for alignment.
BILINGUAL_TXT_PATH="${DIR_SAVE}/${DIRECTION_PAIR}.txt"
echo "Step 1: Preparing bilingual text file at ${BILINGUAL_TXT_PATH}"
python ./metrics/winoMT/prepare_winoST_data.py \
  --input-jsonl $MANIFEST \
  --output-jsonl $INPUT_JSONL \
  --txt-out "${BILINGUAL_TXT_PATH}"

# --- Run alignments ---
ALIGNED_PATH="${DIR_SAVE}/${DIRECTION_PAIR}.aligned"

if [ $ALIGNMENT_METHOD = "fast_align" ]; then
  # Define FastAlign binary files
  FAST_ALIGN_BASE=""
  echo "Step 2: Generating alignments with fast_align at ${ALIGNED_PATH}"
  "${FAST_ALIGN_BASE}/build/fast_align" -i "${BILINGUAL_TXT_PATH}" -d -o -v > "${ALIGNED_PATH}"

else

  awesome-align \
    --output_file="${ALIGNED_PATH}" \
    --model_name_or_path=bert-base-multilingual-cased \
    --data_file="${BILINGUAL_TXT_PATH}" \
    --extraction 'softmax' \
    --cache_dir ./cache/ \
    --batch_size 32 

fi;

# 4. GENERATE PREDICTIONS FOR EACH DATASET
# --- Loop through each dataset type ---
echo "Step 3: Generating prediction JSONL in ${OUT_FOLDER}"
for ds_name in "${DATASET_NAMES[@]}"; do
  # Construct the full path for the source dataset
  dataset_en_path="${BASE_DATA_PATH}/${ds_name}.txt"
  
  # Construct a clean output filename
  # Example: en -> en_it.pred.csv | en_pro -> en_it.pred_en_pro.csv
  
  out_fn="${OUT_FOLDER}/${DIRECTION_PAIR}"
  mkdir -p "$out_fn"
  echo "  -> Processing ${ds_name} dataset -> ${out_fn}"
  
  python -u ./metrics/winoMT/load_alignments.py \
    --dsp="${dataset_en_path}" \
    --ds="${ds_name}" \
    --bi="${BILINGUAL_TXT_PATH}" \
    --align="${ALIGNED_PATH}" \
    --lang="${TARGET_LANG}" \
    --out="${out_fn}" \
    --sys_name="${NAME_SYSTEM}" \
    --ds_name="${ds_name}" \
    --morph="spacy" \
    --batch_size=1 \
    --debug
done

echo "--- [SUCCESS] Finished processing for system: ${NAME_SYSTEM} ---"
