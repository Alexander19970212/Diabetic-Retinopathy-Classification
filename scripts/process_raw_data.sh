#! /bin/bash


DATA_DIR="datasets_raw"
PROCESSED_DIR="datasets"
MAX_SIZE=512
CUT_MODE="max"
TEST_SIZE=0.15
VAL_SIZE=0.15
NUM_PROCESSES=8
declare -i RANDOM_STATE=0xC0FFEE  # to preserve hexadecimal format


DATASETS=("IDRiD" "Messidor" "FGADR" "APTOS" "DDR")


for dataset in "${DATASETS[@]}"; do
    ./data/process.py --image-folder "$DATA_DIR/$dataset" \
                      --output-folder "$PROCESSED_DIR/$dataset" \
                      --max-size $MAX_SIZE --cut-mode $CUT_MODE \
                      --test-size $TEST_SIZE --val-size $VAL_SIZE \
                      --random-state $RANDOM_STATE --num-processes $NUM_PROCESSES
done
