"""
Pack count data file for training/testing.

Input info file entry:
    {
        'image_id': int, MS-COCO image ID.
        'category': int, 0-based category ID.
        'number': int, label.
    }

Input feature file entry:
    {
        'image': string. Path to the image file.
        'boxes': numpy.ndarray, dtype: int16, shape: (N, 4), 
                 N is the number of boxes. 4-dimensions  are (x1, y1, x2, y2).
        'categories': numpy.ndarray, dtype: float32, shape: (N). 
                      1-based category ID.
        'scores': numpy.ndarray, dtype: float32, shape: (N). Detector score.
        '{feature_layer_name}': numpy.ndarray, dtype: float32, shape: (N, D), 
                                D is the feature dimension.
    }

Output format:
    {
        'input': numpy.ndarray, dtype: float32, shape: (N1, D1). 
                 N1 is the number of questions. D1 = 1 + N x (4 + 1 + 1 + D). 
                 N is the number of boxes, D is the feature dimension.
                 Single entry looks like:
                 [cat, ...
                  x_11, y_11, x_12, y_12, cat_1, score_1, feature_1, ...
                  x_21, y_21, x_22, y_22, cat_2, score_2, feature_2, ...
                  ...
                  x_n1, y_n1, x_n2, y_n2, cat_n, score_n, feature_n, ...]
        'label': numpy.ndarray, dtype: int16, shape: (N1).
    }

Usage:
    >> python mscoco_pack_count_data.py -input_info {count info data} \
                                        -input_feature {box feature data} \
                                        -datadir {MS-COCO data folder} \
                                        -output {output file name} \
                                        -num_shards {output number of shards}

Example:
    >> python mscoco_pack_count_data.py -input_info info_?????-of-00001.h5 \
                                        -input_feature feat_?????-of-00009.h5 \
                                        -datadir ../data/mscoco \
                                        -output ../ \
                                        -num_shards {output number of shards}

"""

from data_api import MSCOCO
from utils import logger
from utils.sharded_hdf5 import ShardedFile, ShardedFileReader, ShardedFileWriter
from utils import progress_bar
import numpy as np
import itertools

log = logger.get()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Generate counting questions')
    parser.add_argument('-input_info', help='Count info data file pattern')
    parser.add_argument('-input_feature',
                        help='Bounding boxes and features data file pattern')
    parser.add_argument('-datadir', default='../data/mscoco',
                        help='MS-COCO data folder')
    parser.add_argument('-set', default='train', help='Train/valid')
    parser.add_argument('-output', default=None, help='Output file name')
    parser.add_argument('-num_ex_per_shards', default=10000,
                        help='Number of output example per shard')
    parser.add_argument('-local_feat', default=None,
                        help='Local feature layer name')
    args = parser.parse_args()

    return args


def pack_data(mscoco, info_file, feature_file, local_feat, output_fname, num_ex_per_shards):
    """Pack data together"""
    inps = []
    with ShardedFileReader(info_file) as info_reader:
        num_obj = len(info_reader)
        pb = progress_bar.get(num_obj)
        num_shards = np.ceil(num_obj / float(num_ex_per_shards))
        output_file = ShardedFile(output_fname, num_shards=num_shards)
        with ShardedFileReader(feature_file) as feature_reader:
            with ShardedFileWriter(output_file, num_objects=num_obj) as writer:
                for i, question_entry in itertools.izip(writer, info_reader):
                    image_id = question_entry['image_id']
                    image_path = mscoco.get_image_path(image_id)
                    info = info_reader[image_id]
                    feature = feature_reader[image_path]
                    if local_feat:
                        local_feat_dim = 0
                    else:
                        local_feat_dim = feature[local_feat].shape[-1]
                    num_boxes = feature['boxes'].shape[0]
                    inp_dim = num_boxes * (4 + 1 + 1 + local_feat_dim)
                    inp = np.zeros((num_boxes, inp_dim))
                    for box_i in xrange(num_boxes):
                        inp[box_i, :4] = feature['boxes'][box_i]
                        inp[box_i, 4] = feature['categories'][box_i]
                        inp[box_i, 5] = feature['scores'][box_i]
                        if local_feat:
                            inp[box_i, 6:] = feature[local_feat]
                    inp = inp.reshape(num_boxes * inp_dim)
                    cat = np.array([info['category']])
                    total_inp = np.concatenate((cat, inp)).astype('float32')
                    data = {
                        'input': total_inp
                        'label': info['number']
                    }
                    writer.write(data)
                    pb.increment()


if __name__ == '__main__':
    args = parse_args()
    log.log_args()

    log.info('Info file: {}'.format(args.input_info))
    info_file = ShardedFile.from_pattern_read(args.input_info)

    log.info('Feature file: {}'.format(args.input_feature))
    feature_file = ShardedFile.from_pattern_read(args.input_feature)

    mscoco = MSCOCO(base_dir=args.datadir, set=args.set)
    pack_data(mscoco, info_file, feature_file, args.local_feat,
              args.output, args.num_ex_per_shards)
