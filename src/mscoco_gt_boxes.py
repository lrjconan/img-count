"""
Get groundtruth bounding boxes in MS-COCO.

Usage:
    python mscoco_gt_boxes.py -image_list {image list} \
                              -output {output file}

Example:
    python mscoco_gt_boxes.py -image_list mylist.txt \
                              -output boxes.npy
"""

from data_api import MSCOCO
from utils import list_reader
from utils import logger
from utils import progress_bar
from utils.sharded_hdf5 import ShardedFile, ShardedFileWriter
import argparse
import numpy
import os
import paths

log = logger.get()


def run(mscoco, image_list, output_fname, num_shards):
    """Run all images.

    Args:
        mscoco: MSCOCO API object.
        image_list: list, list of image IDs.
    """
    not_found = []
    cat_rev_dict = mscoco.get_cat_list_reverse()
    fout = ShardedFile(output_fname, num_shards=num_shards)
    pb = progress_bar.get(len(image_list))

    log.info('Running through all images')
    with ShardedFileWriter(fout, num_objects=len(image_list)) as writer:
        for i in writer:
            image_fname = image_list[i]
            image_id = mscoco.get_image_id_from_path(image_fname)
            anns = mscoco.get_image_annotations(image_id)

            if anns is None:
                not_found.append(image_id)
                continue

            num_ann = len(anns)
            boxes = numpy.zeros((num_ann, 4), dtype='int16')
            cats = numpy.zeros((num_ann, 1), dtype='int16')

            for i, ann in enumerate(anns):
                boxes[i, :] = numpy.floor(ann['bbox']).astype('int16')
                cats[i] = cat_rev_dict[ann['category_id']]

            data = {'boxes': boxes, 'categories': cats, 'image': image_fname}
            writer.write(data, key=image_fname)
            pb.increment()

    for image_id in not_found:
        log.error('Not found annotation for image {}'.format(image_id))
    log.error('Total {:d} image annotations not found.'.format(len(not_found)))

    pass


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Get groundtruth boxes in MS-COCO')
    parser.add_argument(
        '-image_list',
        default=os.path.join(paths.GOBI_MREN_MSCOCO, 'image_list_train.txt'),
        help='Image list text file')
    parser.add_argument(
        '-output',
        default=os.path.join(paths.GOBI_MREN_MSCOCO, 'gt_boxes_train.npy'),
        help='Output file name')
    parser.add_argument(
        '-num_shards',
        default=1,
        type=int,
        help='Number of output shards')
    parser.add_argument(
        '-set',
        default='train',
        help='Train/valid set')
    parser.add_argument(
        '-datadir',
        default='../data/mscoco',
        help='Path to MS-COCO dataset')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    log.info('Input list: {0}'.format(args.image_list))
    log.info('Output file: {0}'.format(args.output))
    image_list = list_reader.read_file_list(args.image_list, check=True)
    mscoco = MSCOCO(base_dir=args.datadir, set_name=args.set)
    boxes = run(mscoco, image_list, args.output, args.num_shards)

    pass
