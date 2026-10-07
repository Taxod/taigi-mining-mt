#!/bin/bash
#cd home_test/UnsupPSE # TO CHANGE: folder path

pip install sentencepiece
pip install sentence-transformers # For LaBSE
pip install protobuf==3.20.0 # For glot500
#pip install gensim # When using latest conda environment. WARNING: scipy has been downgraded when installing gensim!
#pip install tokenizers -U

#source ./environment.sh
source ./environment_full_xlmr.sh
export EMBEDDINGS=$RESULTS/new_embeddings
export DOC_EMBEDDINGS=$EMBEDDINGS/doc
# TO CHANGE: model name prefix
# Do not forget the . after the model name for all models except the base one
# ================== custom model==================
export MODEL_PATH=''
PREFIX='pretrained.'
MODEL_ARG='pretrained'
#====================================

#PREFIX='LaBSE.'  #'laser3.' #'glot500.' #'laser.' #'pretrained-hsb.' #'glot500.' #'LaBSE.'
#MODEL_ARG='LaBSE' #  'LaBSE', 'xlmr'
# ================= MKDIR =====================
for src_lang in $SRC_LANGS; do
	for trg_lang in $TRG_LANGS; do
	  case "$src_lang-$trg_lang" in
                han-zh|tl-zh|tl-han|tl-en|poj-zh|poj-en) ;;
                *) continue ;;
            esac
		mkdir -p $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang
		mkdir -p $MINING/bucc2017/$src_lang-$trg_lang
	done;
done;
mkdir -p $DICTIONARIES



## ================= AVERAGED DOCUMENT REPRESENTATION ===============
#With XLM-R representations directly (not tokenised and true cased)
#for data in $BUCC_SETS; do
#    for src_lang in $SRC_LANGS; do
#        for trg_lang in $TRG_LANGS; do
#          case "$src_lang-$trg_lang" in
#                han-zh|tl-zh|tl-han|tl-en|poj-zh|poj-en) ;;
#                *) continue ;;
#            esac
#            for lang in $src_lang $trg_lang; do
#		         echo "$src_lang-$trg_lang sentence embeddings";
#                $PYTHON contextual_sentence_embeddings_NFC.py --input_file $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang \
#                --output_file $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.$lang.vec -m $MODEL_ARG
#            done;
#        done;
#    done;
#done;

## CBIE mode
# ================= APPLY CBIE ===================================
# Run the CBIE python script to transform the generated embeddings
for data in $BUCC_SETS; do
    export DATA_SET=$data
    for src_lang in $SRC_LANGS; do
        for trg_lang in $TRG_LANGS; do
            case "$src_lang-$trg_lang" in
                han-zh|tl-zh|tl-han|tl-en|poj-zh|poj-en) ;;
                *) continue ;;
            esac

            MODEL_NAME=${PREFIX%.}
            echo "Applying CBIE for $src_lang-$trg_lang ($data) using model $MODEL_NAME";
            $PYTHON cbie_transformation.py -m $MODEL_ARG -s $src_lang -t $trg_lang
        done;
    done;
done;

# ================= CBIE DOC_DICTIONARY_GENERATION & MINING =======================
for data in $BUCC_SETS; do

    for src_lang in $SRC_LANGS; do
        for trg_lang in $TRG_LANGS; do
          case "$src_lang-$trg_lang" in
                han-zh|tl-zh|tl-han|tl-en|poj-zh|poj-en) ;;
                *) continue ;;
            esac
            MODEL_NAME=${PREFIX%.}
            CBIE_PREFIX="CBIE2_${MODEL_NAME}."

            # 統一定義相似度檔案的路徑 (對應 PaSeMiLL 文件中的 output_dictionary.sim)
            SIM_FILE=$MINING/bucc2017/$src_lang-$trg_lang/$CBIE_PREFIX$src_lang-$trg_lang.$data.sim

            echo "Step 2: Nearest neighbour mining with CBIE embeddings for $src_lang-$trg_lang";
            $PYTHON scripts/bilingual_nearest_neighbor.py \
                --source_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$CBIE_PREFIX$src_lang-$trg_lang.$data.$src_lang.vec \
                --target_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$CBIE_PREFIX$src_lang-$trg_lang.$data.$trg_lang.vec \
                --output $SIM_FILE \
                --knn 10 -m csls --cslsknn 20 --gpu 0

            # ================= FILTERING_&_EVALUATION ==============================
            for filter_method in $FILTER_METHODS; do
                # 定義過濾後的輸出檔名
                PRED_FILE=$MINING/bucc2017/$src_lang-$trg_lang/${filter_method}_$CBIE_PREFIX$src_lang-$trg_lang.$data.sim.pred

                echo "Step 3: Filtering mined pairs ($filter_method)";
                $PYTHON ./scripts/filter.py \
                    -i $SIM_FILE \
                    -m $filter_method \
                    -th ${FILTER_THRESHOLDS[bucc17_maxalign_${filter_method}_${src_lang}_${trg_lang}]} \
                    -o $PRED_FILE

                echo "Evaluating F-score...";
                $PYTHON scripts/bucc_f-score.py \
                    -p $PRED_FILE \
                    -g $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.gold \
                    > ${PRED_FILE}.res
            done;
        done;
    done;
done;





####### noCBIE process #################
#### ================= DOC_DICTIONARY_GENERATION & MINING =======================
#for data in $BUCC_SETS; do
#    for src_lang in $SRC_LANGS; do
#        for trg_lang in $TRG_LANGS; do
#case "$src_lang-$trg_lang" in
#                han-zh|tl-zh|tl-han|tl-en|poj-zh|poj-en) ;;
#                *) continue ;;
#            esac
#            SIM_FILE=$MINING/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.sim
#
#            echo "Step 2: Nearest neighbour mining for $src_lang-$trg_lang";
#            $PYTHON scripts/bilingual_nearest_neighbor.py \
#                --source_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.$src_lang.vec \
#                --target_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.$trg_lang.vec \
#                --output $SIM_FILE \
#                --knn 10 -m csls --cslsknn 20 --gpu 0
##
##            # ================= FILTERING_&_EVALUATION ==============================
#            for filter_method in $FILTER_METHODS; do
#                # 定義過濾後的輸出檔名
#                PRED_FILE=$MINING/bucc2017/$src_lang-$trg_lang/${filter_method}_$PREFIX$src_lang-$trg_lang.$data.sim.pred
#
#                echo "Step 3: Filtering mined pairs ($filter_method)";
#                $PYTHON ./scripts/filter.py \
#                    -i $SIM_FILE \
#                    -m $filter_method \
#                    -th ${FILTER_THRESHOLDS[bucc17_maxalign_${filter_method}_${src_lang}_${trg_lang}]} \
#                    -o $PRED_FILE
#
#                echo "Evaluating F-score...";
#                $PYTHON scripts/bucc_f-score.py \
#                    -p $PRED_FILE \
#                    -g $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.gold \
#                    > ${PRED_FILE}.res
#            done;
#        done;
#    done;
#done;

echo "Mining fininshed!";