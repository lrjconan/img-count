###############################################################################
# This script will run recurrent instance segmentation on CVPPP dataset.
# It contains 4 stages.
# Stage 1: pretrain patch CNN model
# Stage 2: pretrain box detector model
# Stage 3: train entire network, with box net fixed
# Stage 4: train entire network, finetune
# Stage 5: evaluate the results
###############################################################################
set -e

###############################################################################
# Device
###############################################################################
GPU=$1

###############################################################################
# Folder
###############################################################################
ROOT_FOLDER="/u/mren/code/img-count"
LOG_FOLDER="/u/mren/public_html/results"
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

ATTN_BOX_PADDING_RATIO="0.25"
GT_BOX_CTR_NOISE="0.15"
GT_BOX_PAD_NOISE="0.1"
GT_SEGM_NOISE="0.3"

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

BATCH_SIZE="5"

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
NUM_SAMPLES_PLOT="5"

#NUM_ITER_PATCH="5000"
#NUM_ITER_BOX="20000"
#NUM_ITER_FINETUNE_PATCH="20000"
#NUM_ITER_FINETUNE_TOTAL="20000"
NUM_ITER_PATCH="1"
NUM_ITER_BOX="1"
NUM_ITER_FINETUNE_PATCH="1"
NUM_ITER_FINETUNE_TOTAL="1"

RUN_STAGE10=true
RUN_STAGE15=true
RUN_STAGE20=true
RUN_STAGE25=true
RUN_STAGE30=true
RUN_STAGE35=true
RUN_STAGE40=true
RUN_STAGE50=true

###############################################################################
# Run
###############################################################################
MODEL_ID=$(python src/assign_model_id.py)
source $ROOT_FOLDER/scripts/run_ris.sh

