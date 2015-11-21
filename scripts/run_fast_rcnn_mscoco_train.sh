LOGTO="/ais/gobi3/u/mren/log/fast_rcnn_api" python src/fast_rcnn_api.py \
    -image_list "/ais/gobi3/u/mren/data/mscoco/image_list_train.txt" \
    -dataset "mscoco" \
    -datadir "data/mscoco" \
    -net "lib/fast-rcnn/models/VGG16/coco/test.prototxt" \
    -weights "lib/fast-rcnn/data/fast_rcnn_models/coco_vgg16_fast_rcnn_iter_240000.caffemodel" \
    -gpu $1 \
    -conf 0.3 \
    -nms 0.8 \
    -local_feat \
    -feat_layer "fc7" \
    -proposal "/ais/gobi3/u/mren/data/mscoco/select_search_train.npy" \
    -output "/ais/gobi3/u/mren/data/mscoco/fast_rcnn/fast_rcnn_c3n8_train" \
    -num_images_per_shard 10000

