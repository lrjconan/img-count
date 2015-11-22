LOGTO="/ais/gobi3/u/mren/log/mscoco_gt_boxes" \
    python mscoco_gt_boxes.py \
    -set train \
    -list /ais/gobi3/u/mren/data/mscoco/image_list_train.txt \
    -out /ais/gobi3/u/mren/data/mscoco/gt_boxes_train.npy

LOGTO="/ais/gobi3/u/mren/log/mscoco_gt_boxes" \
    python mscoco_gt_boxes.py \
    -set valid \
    -list /ais/gobi3/u/mren/data/mscoco/image_list_valid.txt \
    -out /ais/gobi3/u/mren/data/mscoco/gt_boxes_valid.npy

