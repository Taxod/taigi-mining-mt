

# Code for the master's thesis "Parallel Sentence Mining and Machine Translation for Low-Resource Taiwanese Hokkien" 

## Introduction
### This repo cover these topics:
1. A parallel sentence mining benchmark
2. Continued pre-training (CPT) of XLM-R base following the Glot500
3. Large-scale mining of POJ–English sentence pairs from Wikipedia
4. Fine-tuning and NLLB-200-distilled-600M and Qwen3.5-0.8B on the mined data
5. Translate sentences with models
6. Evaluation of NLLB-200-distilled-600M and Qwen3.5-0.8B

### This repo use part of the code from the following repos:
[PaSeMiLL](https://github.com/shuokabe/PaSeMiLL): for mining benchmark and large-scale mining task.

[Glot500](https://github.com/cisnlp/Glot500): for continued pre-training (CPT) of XLM-R base

[Outliers](https://github.com/kathyhaem/outliers): draw t-SNE plot

[PARME](https://github.com/DOLMA-NLP/PARME): fine-tune NLLB-200-distilled-600M

[Belopsem](https://github.com/shuokabe/Belopsem/): preprocessing and data spliting

### This thesis use the tool developed from the following repos:
[taigi-converter](https://github.com/taigikeyboard/taigi-converter): for the conversion between POJ and TL

[hanzidentifier](https://github.com/tsroten/hanzidentifier):
for identified simplified Chinese and traditional Chinese


## 1. Parallel Sentence Mining Benchmark:
```bash
bash ./PaSEMiLL/code/mine_bucc_full_xlmr_new.sh
```

## 2. Continued pre-training (CPT) of XLM-R base following the Glot500
```bash
# dataset are from Glot500-c
bash ./Glot500/run_glot500.sh
```

## 3. Large-scale mining of POJ–English sentence pairs from Wikipedia
```bash 
bash ./scripts/run_pasemill.sh
```

## 4. Fine-tuning and NLLB-200-distilled-600M and Qwen3.5-0.8B on the mined data
### fine-tuen NLLB-200-distilled-600M
```bash
python ./PARME/codes/build_datasets.py \
    --poj-file /mining_training/data/mined_new/nan_eng/CP_wikiplus10/output_poj_th_0.80.txt \
    --eng-file /mining_training/data/mined_new/nan_eng/CP_wikiplus10/output_eng_th_0.80.txt \
    --out-dir /mining_training/data/mined_new/nan_eng/nllb_CPT_080
python ./PARME/codes/run_translation.py
bash ./PARME/nllb.sh
```

### fine-tune Qwen3.5-0.8B
```bash
#fine-tune CPT mined data on Qwen3.5-0.8B
llamafactory-cli train configs/qwen_sft.yaml

# fine-tune LaBSE mined data on Qwen3.5-0.8B
llamafactory-cli train configs/qwen_sft.yaml

# fine-tune simple match data on Qwen3.5-0.8B
llamafactory-cli train configs/qwen_sft.yaml
```
## 5. Translate sentences with models
```bash

cd translation
# NLLB
python nllb_translation.py

# Qwen
python qwen_translation.py

# Gemini
python gemini_openrouter.py

# Taigi-llama model
!python taigi_llama_translation.py \
  --input    input/raw_poj_en/poj-en.poj \
  --ref-file input/raw_poj_en/poj-en.en \
  --output   "/llama_tra/poj_en_translation_7b.csv" \
  --model-dir Bohanlu/Taigi-Llama-2-Translator-7B \
  --target-lang EN \
  --batch-size 2 --block-size 128 --max-new-tokens 256 --dtype float16
```

## 6. Evaluation of NLLB-200-distilled-600M and Qwen3.5-0.8B
```bash
# example for taigi-llama
USE_TF=0 USE_FLAX=0 python score.py \
  --pattern "/llama_chat/*_translation_*.csv" \
  --model-path "llama_chat" \
  --prompt-mode "llama" \
  --xcomet-batch-size 8
  
# analyze xCOMET-XL error span output language and repetition
python ./scripts/analysis_translation.py
```

## draw t-SNE plot
```bash
python extract_sent_embeddings.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --dataset custom_parallel \
    --custom_files file1 file2 file3

python ./vis_tsne.py \
    --emb_files ../embs/custom/all-mpnet-base-v2/7/lang1.pt ../embs/custom/all-mpnet-base-v2/7/lang2.pt ../embs/custom/all-mpnet-base-v2/7/lang3.pt \
    --labels "1" "2" "3" \
    --plot_file ./apod_all-mpnet-base-v2_domain_investigation.png
```


## Retrive trdirtional chinese Glot500-c datasets 
```bash
python ./scripts/haszidentifer.py
```