###############################################################################
# Device
###############################################################################
if [-n "$GPU" ]; then
    echo "Running on GPU $GPU"
else
    echo "Running on CPU"
    GPU="-1"
fi

###############################################################################
# Model IDs
###############################################################################
PATCH_MODEL_ID="ris_patch_"$DATASET"-"$MODEL_ID
BOX_MODEL_ID="ris_box_"$DATASET"-"$MODEL_ID
FINETUNE_PATCH_MODEL_ID="ris_ft0_"$DATASET"-"$MODEL_ID
FINETUNE_TOTAL_MODEL_ID="ris_ft1_"$DATASET"-"$MODEL_ID

echo "Stage 1: "$PATCH_MODEL_ID
echo "Stage 2: "$BOX_MODEL_ID
echo "Stage 3: "$FINETUNE_PATCH_MODEL_ID
echo "Stage 4: "$FINETUNE_TOTAL_MODEL_ID

###############################################################################
# Inter-stage weights
###############################################################################
PATCH_WEIGHTS=$SAVE_FOLDER/$PATCH_MODEL_ID/weights.h5
BOX_WEIGHTS=$SAVE_FOLDER/$BOX_MODEL_ID/weights.h5
FINETUNE_PATCH_WEIGHTS=$SAVE_FOLDER/$FINETUNE_PATCH_MODEL_ID/weights.h5

###############################################################################
# Program paths
###############################################################################
RIS_PATCH_TRAIN_PROG=$ROOT_FOLDER/src/ris_patch.py
RIS_PATCH_READ_PROG=$ROOT_FOLDER/src/ris_patch_reader.py
RIS_BOX_TRAIN_PROG=$ROOT_FOLDER/src/ris_box.py
RIS_BOX_READ_PROG=$ROOT_FOLDER/src/ris_box_reader.py
RIS_TRAIN_PROG=$ROOT_FOLDER/src/ris.py
RIS_READ_PROG=$ROOT_FOLDER/src/ris_reader.py
RIS_EVAL_PROG=$ROOT_FOLDER/src/ris_eval.py

###############################################################################
# Stage 1.0
###############################################################################
if [ "$RUN_STAGE10" = true ]; then
    echo "Stage 1: training patch net"
    python $RIS_PATCH_TRAIN_PROG \
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
    --attn_box_padding_ratio $ATTN_BOX_PADDING_RATIO \
    --gt_box_ctr_noise $GT_BOX_CTR_NOISE \
    --gt_box_pad_noise $GT_BOX_PAD_NOISE \
    --gt_segm_noise $GT_SEGM_NOISE \
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
    --base_learn_rate $BASE_LEARN_RATE_STAGE1 \
    --batch_size $BATCH_SIZE \
    --fixed_order \
    --num_steps $NUM_ITER_PATCH \
    --model_id $PATCH_MODEL_ID
fi

###############################################################################
# Stage 1.5
###############################################################################
if [ "$RUN_STAGE15" = true ]; then
    echo "Reading weights from stage 1"
    python $RIS_PATCH_READ_PROG \
    --model_id $PATCH_MODEL_ID \
    --results $SAVE_FOLDER \
    --output $PATCH_WEIGHTS
fi

###############################################################################
# Stage 2.0
###############################################################################
if [ "$RUN_STAGE20" = true ]; then
    echo "Stage 2: training box net"
    python $RIS_BOX_TRAIN_PROG \
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
    --base_learn_rate $BASE_LEARN_RATE_STAGE2 \
    --num_steps $NUM_ITER_BOX \
    --model_id $BOX_MODEL_ID
fi

###############################################################################
# Stage 2.5
###############################################################################
if [ "$RUN_STAGE25" = true ]; then
    echo "Reading weights from stage 2"
    python $RIS_BOX_READ_PROG \
    --model_id $BOX_MODEL_ID \
    --results $SAVE_FOLDER \
    --output $BOX_WEIGHTS
fi

###############################################################################
# Stage 3.0
###############################################################################
if [ "$RUN_STAGE30" = true ]; then
    echo "Stage 3: training entire network, with box net fixed"
    python $RIS_TRAIN_PROG \
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
    --freeze_ctrl_cnn \
    --freeze_ctrl_rnn \
    --pretrain_attn_net $PATCH_WEIGHTS \
    --batch_size $BATCH_SIZE \
    --save_ckpt \
    --base_learn_rate $BASE_LEARN_RATE_STAGE3 \
    --num_steps $NUM_ITER_FINETUNE_PATCH \
    --model_id $FINETUNE_PATCH_MODEL_ID
fi

###############################################################################
# Stage 3.5
###############################################################################
if [ "$RUN_STAGE35" = true ]; then
    echo "Reading weights from stage 3"
    python $RIS_READ_PROG \
    --model_id $FINETUNE_PATCH_MODEL_ID \
    --results $SAVE_FOLDER \
    --output $FINETUNE_PATCH_WEIGHTS
fi

###############################################################################
# Stage 4.0
###############################################################################
if [ "$RUN_STAGE40" = true ]; then
    echo "Stage 4: training entire network, finetune"
    python $RIS_TRAIN_PROG \
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
    --knob_box_offset $KNOB_BOX_OFFSET1 \
    --knob_segm_offset $KNOB_SEGM_OFFSET1 \
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
    --freeze_ctrl_cnn \
    --batch_size $BATCH_SIZE \
    --save_ckpt \
    --base_learn_rate $BASE_LEARN_RATE_STAGE4 \
    --num_steps $NUM_ITER_FINETUNE_TOTAL \
    --model_id $FINETUNE_TOTAL_MODEL_ID
fi

###############################################################################
# Stage 5.0
###############################################################################
if [ "$RUN_STAGE50" = true ]; then
    echo "Running evaluation"
    python $RIS_EVAL_PROG \
    --dataset $DATASET \
    --model_id $FINETUNE_TOTAL_MODEL_ID \
    --results $SAVE_FOLDER
fi
