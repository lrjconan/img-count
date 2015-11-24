python src/mscoco_pack_count_data.py \
    -input_info "/ais/gobi3/u/mren/data/coco-count/coco_count_info-?????-of-00001.h5" \
    -input_feature "/ais/gobi3/u/mren/data/mscoco/fast_rcnn/fast_rcnn_train_gt-?????-of-00009.h5" \
    -local_feat "pool5" \
    -datadir "/ais/gobi3/datasets/mscoco" \
    -set "train" \
    -num_ex_per_shard 10000 \
    -output "/ais/gobi3/u/mren/data/coco-count/coco_count_pool5_train"

