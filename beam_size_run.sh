#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define paths and variables
DATA_PATH="data/en-fr/prepared"
SOURCE_LANG="fr"
TARGET_LANG="en"
BEAM_SIZE=23
OUTPUT_DIR="assignments/05/beamsize${BEAM_SIZE}"
TRANSLATION_OUTPUT="${OUTPUT_DIR}/translations.txt"
POSTPROCESSED_OUTPUT="${OUTPUT_DIR}/translations.p.txt"
CHECKPOINT_PATH="assignments/03/baseline/checkpoints/checkpoint_last.pt"
POSTPROCESS_SCRIPT="scripts/postprocess.sh"
RESULTS_FILE="${OUTPUT_DIR}/bleu_results.txt"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Step 1: Translating the dataset..."
python translate_beam.py \
    --data "$DATA_PATH" \
    --dicts "$DATA_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output "$TRANSLATION_OUTPUT" \
    --beam-size "$BEAM_SIZE"

echo "Step 2: Postprocessing translations..."
bash "$POSTPROCESS_SCRIPT" \
    "$TRANSLATION_OUTPUT" \
    "$POSTPROCESSED_OUTPUT" \
    "$TARGET_LANG"

# Evaluate using BLEU score and save results
echo "Step 3: Evaluating translations..."
if command -v sacrebleu &>/dev/null; then
    echo "BLEU Score Evaluation Results - $(date)" | tee -a "$RESULTS_FILE"
    cat "$POSTPROCESSED_OUTPUT" | sacrebleu data/en-fr/raw/test.en | tee -a "$RESULTS_FILE"
else
    echo "Error: sacrebleu is not installed. Please install it to calculate BLEU scores." >&2
    exit 1
fi

echo "Pipeline complete. Postprocessed translations are saved in: $POSTPROCESSED_OUTPUT"
