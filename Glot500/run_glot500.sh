set -e

# ==========================================
# 0. Define Absolute Paths
# ==========================================
BASE_DIR=""
DATA_DIR="${BASE_DIR}/data/"
OUTPUT_DIR=""
GLOT500_REPO="${BASE_DIR}/Glot500"

MERGE_TOK_DIR="${OUTPUT_DIR}/merged_for_tok"
MERGE_LM_DIR="${OUTPUT_DIR}/merged_for_lm"
TOKENIZER_DIR="${OUTPUT_DIR}/trained_tokenizer/"
MODEL_DIR="${OUTPUT_DIR}/model_checkpoints"
CACHE_DIR="${OUTPUT_DIR}/hf_cache"
mkdir -p ${MERGE_TOK_DIR}
mkdir -p ${MERGE_LM_DIR}
mkdir -p ${TOKENIZER_DIR}
mkdir -p ${MODEL_DIR}
mkdir -p ${CACHE_DIR}
# ==========================================
# 1. Environment Activation
# ==========================================
echo ">>> Activating Conda environment..."
source ${BASE_DIR}/miniconda3/etc/profile.d/conda.sh
conda activate glot500_env

export PYTHONUNBUFFERED=1

# ==========================================
# 2. Data Preprocessing (merge_files.py)
# ==========================================
 echo ">>> [1/3] Starting Data Preprocessing..."
 cd ${GLOT500_REPO}/preprocessing

 python merge_files.py \
  --data_directory ${DATA_DIR} \
  --save_directory ${MERGE_TOK_DIR} \
  --experiment_name Glot500 \
  --lg_sampling_factor 0.3 \
  --scale 30

 python merge_files.py \
  --data_directory ${DATA_DIR} \
  --save_directory ${MERGE_LM_DIR} \
  --experiment_name Glot500 \
  --lg_sampling_factor 0.3 \
  --scale 30

# ==========================================
# 3. Tokenizer Training
# ==========================================
 echo ">>> [2/3] Starting Tokenizer Training..."
 cd ${GLOT500_REPO}/tokenization

python run.py \
  --input_fname ${MERGE_TOK_DIR}/Glot500.txt \
  --model_name xlm-roberta-base \
  --save_directory ${TOKENIZER_DIR} \
  --vocab_size 100000

# ==========================================
# 4. Language Model Pre-training
# ==========================================
echo ">>> [3/3] Starting Model Training..."
cd ${GLOT500_REPO}/modeling

WANDB_DISABLED=true  torchrun  --nproc_per_node=1 run.py \
  --model_name_or_path xlm-roberta-base \
  --train_file ${MERGE_LM_DIR}/Glot500.txt \
  --tokenizer_name ${TOKENIZER_DIR}Glot500_extended_spm \
  --output_dir ${MODEL_DIR} \
  --cache_dir ${CACHE_DIR} \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 4 \
  --fp16 True \
  --do_train \
  --num_train_epochs 10 \
  --save_steps 10000 \
  --ddp_timeout 259200

echo ">>> Pipeline Completed Successfully!"