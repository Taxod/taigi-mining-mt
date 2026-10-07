#!/bin/bash
#SBATCH --job-name=pasemill_mining
#SBATCH --output=logs/pasemill_%j.out     # Standard output log
#SBATCH --error=logs/pasemill_%j.err      # Standard error log
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=96G
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
set -eou pipefail
# scripts/run_pasemill.sh

## ==========================================
## Check Hardware and Environment Status
## ==========================================
#echo ">>> Hardware Status Report <<<"
#
#echo "[1] System RAM Usage:"
#free -h
#
##echo "[2] GPU Status:"
##nvidia-smi
#
#echo "[3] Disk Space Quota (Current Directory):"
#df -h .
#
#echo "[4] Slurm Allocated Resources:"
#echo "Allocated CPUs: ${SLURM_CPUS_ON_NODE}"
#echo "Allocated Memory (MB): ${SLURM_MEM_PER_NODE}"
#echo "Node Name: ${SLURMD_NODENAME}"
#echo "=========================================="

# Get absolute path of the project
PROJECT_ROOT=$(pwd)
CLEANED_DIR="${PROJECT_ROOT}/data/cleaned"
EMBED_DIR="${PROJECT_ROOT}/data/embeddings"
MINED_DIR="${PROJECT_ROOT}/data/mined"

# Parameter settings
SRC_LANG="poj"
TRG_LANG="en"
DATA_SET="wiki_plus10"
PREFIX="LaBSE."
MODEL_NAME="LaBSE"
FILTER_METHOD="static" # Can be replaced with your desired filtering method (e.g., fw_bw)
THRESHOLD="0.10142940208653206"       # Adjust threshold based on experiment results

# Feature Toggles
ENABLE_CBIE="false"     # Set to "true" to enable CBIE, "false" to disable

conda activate thesis_env

# 1. Create pseudo-BUCC directory structure expected by PaSeMiLL
mkdir -p "${EMBED_DIR}/bucc2017/${SRC_LANG}-${TRG_LANG}"
mkdir -p "${MINED_DIR}/bucc2017/${SRC_LANG}-${TRG_LANG}"
mkdir -p "${CLEANED_DIR}/bucc2017/${SRC_LANG}-${TRG_LANG}"

# Copy cleaned monolingual data and rename to expected format
cp "${CLEANED_DIR}/nan_wiki_mining.id.txt" "${CLEANED_DIR}/bucc2017/${SRC_LANG}-${TRG_LANG}/${SRC_LANG}-${TRG_LANG}.${DATA_SET}.${SRC_LANG}"
cp "${CLEANED_DIR}/eng_wiki_mining.id.txt" "${CLEANED_DIR}/bucc2017/${SRC_LANG}-${TRG_LANG}/${SRC_LANG}-${TRG_LANG}.${DATA_SET}.${TRG_LANG}"

# Export environment variables for PaSeMiLL scripts
export DOC_EMBEDDINGS="${EMBED_DIR}"
export MINING="${MINED_DIR}"
export DATA="${CLEANED_DIR}"
export PYTHONUNBUFFERED=1
export DATA_SET="${DATA_SET}"
export MODEL_PATH="${PROJECT_ROOT}/models/checkpoint-690000"
export INPUT_PREFIX="${PREFIX}"
export OUTPUT_PREFIX="CBIE2_${PREFIX}"
echo ">>> Switching to PaSeMiLL directory to execute algorithms..."
cd PaSeMiLL

## 2. Contextual Sentence Embeddings
for lang in ${SRC_LANG} ${TRG_LANG}; do
    echo ">> Generating embeddings for ${lang}..."
    stdbuf -oL -eL python code/contextual_sentence_embeddings_new.py \
        --input_file "${DATA}/bucc2017/${SRC_LANG}-${TRG_LANG}/${SRC_LANG}-${TRG_LANG}.${DATA_SET}.${lang}" \
        --output_file "${DOC_EMBEDDINGS}/bucc2017/${SRC_LANG}-${TRG_LANG}/${PREFIX}${SRC_LANG}-${TRG_LANG}.${DATA_SET}.${lang}.vec" \
        -m "${MODEL_NAME}" \
#        -mp "${PROJECT_ROOT}/models/checkpoint-690000"
done

# 3. CBIE Transformation (Isotropic Enhancement)
if [ "${ENABLE_CBIE}" = "true" ]; then
    echo ">> Applying CBIE..."
    stdbuf -oL -eL python code/cbie_transformation_new.py -m "${MODEL_NAME}" -s ${SRC_LANG} -t ${TRG_LANG}
    CURRENT_PREFIX="${OUTPUT_PREFIX}"
else
    echo ">> Skipping CBIE..."
    CURRENT_PREFIX="${PREFIX}"
fi

# 4. Nearest Neighbour Mining (Bilingual alignment mining)
SIM_FILE="${MINING}/bucc2017/${SRC_LANG}-${TRG_LANG}/${CURRENT_PREFIX}${SRC_LANG}-${TRG_LANG}.${DATA_SET}.sim"
echo ">> Running nearest neighbour mining..."
KNN_GPU_ARG=""
if command -v nvidia-smi &> /dev/null; then
    echo ">> NVIDIA GPU detected. Enabling --gpu 0 for K-NN search."
    KNN_GPU_ARG="--gpu 0"
else
    echo ">> No NVIDIA GPU detected. Running K-NN search on default hardware."
fi

python code/scripts/bilingual_nearest_neighbor.py \
    --source_embeddings "${DOC_EMBEDDINGS}/bucc2017/${SRC_LANG}-${TRG_LANG}/${CURRENT_PREFIX}${SRC_LANG}-${TRG_LANG}.${DATA_SET}.${SRC_LANG}.vec" \
    --target_embeddings "${DOC_EMBEDDINGS}/bucc2017/${SRC_LANG}-${TRG_LANG}/${CURRENT_PREFIX}${SRC_LANG}-${TRG_LANG}.${DATA_SET}.${TRG_LANG}.vec" \
    --output "${SIM_FILE}" \
    --knn 10 -m csls --cslsknn 20 ${KNN_GPU_ARG}

# 5. Filtering mined parallel sentences
PRED_FILE="${MINING}/bucc2017/${SRC_LANG}-${TRG_LANG}/${FILTER_METHOD}_${CURRENT_PREFIX}${SRC_LANG}-${TRG_LANG}.${DATA_SET}.sim.pred"
echo ">> Filtering mined pairs using threshold ${THRESHOLD}..."
python code/scripts/filter.py \
    -i "${SIM_FILE}" \
    -m ${FILTER_METHOD} \
    -th ${THRESHOLD} \
    -o "${PRED_FILE}"

echo ">>> Returning to project root directory and extracting final parallel corpus..."
cd "${PROJECT_ROOT}"
cp "${PRED_FILE}" "${MINED_DIR}/aligned.tsv"
echo "Mining Pipeline Completed Successfully!"