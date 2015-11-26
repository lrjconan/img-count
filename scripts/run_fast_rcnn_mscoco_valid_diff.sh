for CONF in 0.7 0.8 0.9
do
    for NMS in 0.2 0.3 0.4
    do
        echo "CONF: $CONF"
        echo "NMS: $NMS"
        LOGTO="/ais/gobi3/u/mren/log/fast_rcnn_api" \
            python src/fast_rcnn_api.py \
            -gpu $1 \
            -image_list "/ais/gobi3/u/mren/data/mscoco/image_list_valid.txt" \
            -dataset "mscoco" \
            -datadir "data/mscoco" \
            -net "lib/fast-rcnn/models/VGG16/coco/test.prototxt" \
            -weights "lib/fast-rcnn/data/fast_rcnn_models/coco_vgg16_fast_rcnn_iter_240000.caffemodel" \
            -conf $CONF \
            -nms $NMS \
            -local_feat "fc7,pool5" \
            -proposal "/ais/gobi3/u/mren/data/mscoco/select_search_valid.npy" \
            -output "/ais/gobi3/u/mren/data/mscoco/fast_rcnn/fast_rcnn_valid_conf_""$CONF""_nms_""$NMS" \
            -num_images_per_shard 5000 \
            -shuffle \
            -max_num_images 5000
    done
done
