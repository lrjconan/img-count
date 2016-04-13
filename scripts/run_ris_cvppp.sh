###############################################################################
# This script will run recurrent instance segmentation on CVPPP dataset.
# It contains 4 stages.
# Stage 1: pretrain patch CNN model
# Stage 2: pretrain box detector model
# Stage 3: train entire network, with box net fixed
# Stage 4: train entire network, finetune
# Final stage: evaluate the results
###############################################################################
set -e

###############################################################################
# Device
###############################################################################
GPU=$1

###############################################################################
# Folder
LOG_FOLDER="/u/mren/public_html/results/"
SAVE_FOLDER="/ais/gobi3/u/mren/results/img-count-pipeline"
LOCALHOST="www.cs.toronto.edu/~mren"

###############################################################################
# Dataset
###############################################################################
DATASET="cvppp"
DATASET_FOLDER="/ais/gobi3/u/mren/data/lsc/A1"

###############################################################################
# Hyper parameters
###############################################################################
CTRL_CNN_FILTER_SIZE="3,3,3,3,3,3,3,3"
CTRL_CNN_DEPTH="8,8,16,16,32,32,64,64"
CTRL_CNN_POOL="1,2,1,2,1,2,2,2"
NUM_CTRL_MLP_LAYERS="1"

ATTN_CNN_FILTER_SIZE="3,3,3,3,3,3"
ATTN_CNN_DEPTH="8,8,16,16,32,32"
ATTN_CNN_POOL="1,2,1,2,1,2"

NUM_ATTN_MLP_LAYERS="1"
ATTN_MLP_DEPTH="16"

ATTN_DCNN_FILTER_SIZE="3,3,3,3,3,3,3"
ATTN_DCNN_DEPTH="32,32,16,16,8,8,1"
ATTN_DCNN_POOL="2,1,2,1,2,1,1"

FILTER_HEIGHT="48"
FILTER_WIDTH="48"

SEGM_LOSS_FN="iou"
BOX_LOSS_FN="iou"

CTRL_RNN_INP_STRUCT="attn"

BATCH_SIZE="4"

KNOB_DECAY="0.5"
KNOB_BOX_OFFSET0="-50000"
KNOB_SEGM_OFFSET0="3000"
STEPS_PER_KNOB_DECAY="700"
KNOB_BOX_OFFSET1="-50000"
KNOB_SEGM_OFFSET1="-50000"

BASE_LEARN_RATE_STAGE1="0.001"
BASE_LEARN_RATE_STAGE2="0.001"
BASE_LEARN_RATE_STAGE3="0.001"
BASE_LEARN_RATE_STAGE4="0.0005"

###############################################################################
# Training options
###############################################################################
STEPS_PER_VALID="50"
STEPS_PER_PLOT="500"
STEPS_PER_TRAINVAL="50"
STEPS_PER_LOG="10"
NUM_SAMPLES_PLOT="20"

# NUM_ITER_PATCH="10000"
# NUM_ITER_BOX="30000"
# NUM_ITER_FINETUNE_PATCH="30000"
# NUM_ITER_FINETUNE_TOTAL="30000"
NUM_ITER_PATCH="20"
NUM_ITER_BOX="20"
NUM_ITER_FINETUNE_PATCH="20"
NUM_ITER_FINETUNE_TOTAL="20"

# Model IDs
MODEL_ID=$(python src/assign_model_id.py)
PATCH_MODEL_ID="ris_patch-"$MODEL_ID
BOX_MODEL_ID="ris_box-"$MODEL_ID
FINETUNE_PATCH_MODEL_ID="ris_ft0-"$MODEL_ID
FINETUNE_TOTAL_MODEL_ID="ris_ft1-"$MODEL_ID
echo $PATCH_MODEL_ID
echo $BOX_MODEL_ID
echo $FINETUNE_PATCH_MODEL_ID
echo $FINETUNE_TOTAL_MODEL_ID

###############################################################################
# Stage 1
###############################################################################
echo "Stage 1: training patch net"
python src/ris_patch.py \
--gpu $GPU \
--dataset $DATASET \
--dataset_folder $DATASET_FOLDER \
--logs $LOG_FOLDER \
--localhost $LOCALHOST \
--results $SAVE_FOLDER \
--steps_per_valid $STEPS_PER_VALID \
--steps_per_plot $STEPS_PER_PLOT \
--steps_per_trainval $STEPS_PER_TRAINVAL \
--steps_per_log $STEPS_PER_LOG \
--num_samples_plot $NUM_SAMPLES_PLOT \
--attn_cnn_filter_size $ATTN_CNN_FILTER_SIZE \
--attn_cnn_depth $ATTN_CNN_DEPTH \
--attn_cnn_pool $ATTN_CNN_POOL \
--num_attn_mlp_layers $NUM_ATTN_MLP_LAYERS \
--attn_mlp_depth $ATTN_MLP_DEPTH \
--attn_dcnn_filter_size $ATTN_DCNN_FILTER_SIZE \
--attn_dcnn_depth $ATTN_DCNN_DEPTH \
--attn_dcnn_pool $ATTN_DCNN_POOL \
--num_attn_mlp_layers 1 \
--filter_height $FILTER_HEIGHT \
--filter_width $FILTER_WIDTH \
--segm_loss_fn $SEGM_LOSS_FN \
--save_ckpt \
--batch_size $BATCH_SIZE \
--fixed_order \
--num_steps $NUM_ITER_PATCH \
--model_id $PATCH_MODEL_ID

###############################################################################
# Stage 1.5
###############################################################################
echo "Reading weights from stage 1"
PATCH_WEIGHTS=$SAVE_FOLDER/$PATCH_MODEL_ID/weights.h5
python src/ris_patch_reader.py \
--model_id $PATCH_MODEL_ID \
--results $SAVE_FOLDER \
--output $PATCH_WEIGHTS

###############################################################################
# Stage 2
###############################################################################
echo "Stage 2: training box net"
python src/ris_box.py \
--gpu $GPU \
--dataset $DATASET \
--dataset_folder $DATASET_FOLDER \
--logs $LOG_FOLDER \
--localhost $LOCALHOST \
--results $SAVE_FOLDER \
--steps_per_valid $STEPS_PER_VALID \
--steps_per_plot $STEPS_PER_PLOT \
--steps_per_trainval $STEPS_PER_TRAINVAL \
--steps_per_log $STEPS_PER_LOG \
--num_samples_plot $NUM_SAMPLES_PLOT \
--ctrl_cnn_filter_size $CTRL_CNN_FILTER_SIZE \
--ctrl_cnn_depth $CTRL_CNN_DEPTH \
--ctrl_cnn_pool $CTRL_CNN_POOL \
--num_ctrl_mlp_layers $NUM_CTRL_MLP_LAYERS \
--pretrain_cnn $PATCH_WEIGHTS \
--batch_size $BATCH_SIZE \
--ctrl_rnn_inp_struct $CTRL_RNN_INP_STRUCT \
--save_ckpt \
--num_steps $NUM_ITER_BOX \
--model_id $BOX_MODEL_ID

###############################################################################
# Stage 2.5
###############################################################################
echo "Reading weights from stage 2"
BOX_WEIGHTS=$SAVE_FOLDER/$BOX_MODEL_ID/weights.h5
python src/ris_box_reader.py \
--model_id $BOX_MODEL_ID \
--results $SAVE_FOLDER \
--output $BOX_WEIGHTS

###############################################################################
# Stage 3
###############################################################################
echo "Stage 3: training entire network, with box net fixed"
python src/ris.py \
--gpu $GPU \
--dataset $DATASET \
--dataset_folder $DATASET_FOLDER \
--logs $LOG_FOLDER \
--localhost $LOCALHOST \
--results $SAVE_FOLDER \
--steps_per_valid $STEPS_PER_VALID \
--steps_per_plot $STEPS_PER_PLOT \
--steps_per_trainval $STEPS_PER_TRAINVAL \
--steps_per_log $STEPS_PER_LOG \
--num_samples_plot $NUM_SAMPLES_PLOT \
--use_knob \
--knob_decay $KNOB_DECAY \
--steps_per_knob_decay $STEPS_PER_KNOB_DECAY \
--knob_box_offset $KNOB_BOX_OFFSET0 \
--knob_segm_offset $KNOB_SEGM_OFFSET0 \
--knob_use_timescale \
--box_loss_fn $BOX_LOSS_FN \
--segm_loss_fn $SEGM_LOSS_FN \
--ctrl_cnn_filter_size $CTRL_CNN_FILTER_SIZE \
--ctrl_cnn_depth $CTRL_CNN_DEPTH \
--ctrl_cnn_pool $CTRL_CNN_POOL \
--num_ctrl_mlp_layers $NUM_CTRL_MLP_LAYERS \
--ctrl_rnn_inp_struct $CTRL_RNN_INP_STRUCT \
--attn_cnn_filter_size $ATTN_CNN_FILTER_SIZE \
--attn_cnn_depth $ATTN_CNN_DEPTH \
--attn_cnn_pool $ATTN_CNN_POOL \
--num_attn_mlp_layers $NUM_ATTN_MLP_LAYERS \
--attn_mlp_depth $ATTN_MLP_DEPTH \
--attn_dcnn_filter_size $ATTN_DCNN_FILTER_SIZE \
--attn_dcnn_depth $ATTN_DCNN_DEPTH \
--attn_dcnn_pool $ATTN_DCNN_POOL \
--filter_height $FILTER_HEIGHT \
--filter_width $FILTER_WIDTH \
--fixed_gamma \
--pretrain_ctrl_net $BOX_WEIGHTS \
--freeze_ctrl_net \
--pretrain_attn_net $PATCH_WEIGHTS \
--batch_size $BATCH_SIZE \
--save_ckpt \
--num_steps $NUM_ITER_FINETUNE_PATCH \
--model_id $FINETUNE_PATCH_MODEL_ID

###############################################################################
# Stage 3
###############################################################################
echo "Reading weights from stage 3"
FINETUNE_PATCH_WEIGHTS=$SAVE_FOLDER/$FINETUNE_PATCH_MODEL_ID/weights.h5
python src/ris_reader.py \
--model_id $FINETUNE_PATCH_MODEL_ID \
--results $SAVE_FOLDER \
--output $FINETUNE_PATCH_WEIGHTS

###############################################################################
# Stage 4
###############################################################################
echo "Stage 4: training entire network, finetune"
python src/ris.py \
--gpu $GPU \
--dataset $DATASET \
--dataset_folder $DATASET_FOLDER \
--logs $LOG_FOLDER \
--localhost $LOCALHOST \
--results $SAVE_FOLDER \
--steps_per_valid $STEPS_PER_VALID \
--steps_per_plot $STEPS_PER_PLOT \
--steps_per_trainval $STEPS_PER_TRAINVAL \
--steps_per_log $STEPS_PER_LOG \
--num_samples_plot $NUM_SAMPLES_PLOT \
--use_knob \
--knob_decay $KNOB_DECAY \
--steps_per_knob_decay $STEPS_PER_KNOB_DECAY \
--knob_box_offset $KNOB_BOX_OFFSET0 \
--knob_segm_offset $KNOB_SEGM_OFFSET0 \
--knob_use_timescale \
--box_loss_fn $BOX_LOSS_FN \
--segm_loss_fn $SEGM_LOSS_FN \
--ctrl_cnn_filter_size $CTRL_CNN_FILTER_SIZE \
--ctrl_cnn_depth $CTRL_CNN_DEPTH \
--ctrl_cnn_pool $CTRL_CNN_POOL \
--num_ctrl_mlp_layers $NUM_CTRL_MLP_LAYERS \
--ctrl_rnn_inp_struct $CTRL_RNN_INP_STRUCT \
--attn_cnn_filter_size $ATTN_CNN_FILTER_SIZE \
--attn_cnn_depth $ATTN_CNN_DEPTH \
--attn_cnn_pool $ATTN_CNN_POOL \
--num_attn_mlp_layers $NUM_ATTN_MLP_LAYERS \
--attn_mlp_depth $ATTN_MLP_DEPTH \
--attn_dcnn_filter_size $ATTN_DCNN_FILTER_SIZE \
--attn_dcnn_depth $ATTN_DCNN_DEPTH \
--attn_dcnn_pool $ATTN_DCNN_POOL \
--filter_height $FILTER_HEIGHT \
--filter_width $FILTER_WIDTH \
--fixed_gamma \
--pretrain_net $FINETUNE_PATCH_WEIGHTS \
--batch_size $BATCH_SIZE \
--save_ckpt \
--base_learn_rate $BASE_LEARN_RATE_STAGE4 \
--num_steps $NUM_ITER_FINETUNE_TOTAL \
--model_id $FINETUNE_TOTAL_MODEL_ID

###############################################################################
# Stage 4.5
###############################################################################
echo "Reading weights from stage 4"
FINETUNE_TOTAL_WEIGHTS=$SAVE_FOLDER/$FINETUNE_TOTAL_MODEL_ID/weights.h5
python src/ris_reader.py \
--model_id $FINETUNE_TOTAL_MODEL_ID \
--results $SAVE_FOLDER \
--output $FINETUNE_TOTAL_WEIGHTS

###############################################################################
# Stage 5
###############################################################################
echo "Running evaluation"
python src/ris_eval.py \
--dataset $DATASET \
--model_id $FINETUNE_TOTAL_MODEL_ID \
--results $SAVE_FOLDER
