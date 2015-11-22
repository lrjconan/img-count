"""
Get groundtruth bounding boxes in MS-COCO.

Usage:
    python mscoco_gt_boxes.py \
            -list {image list} \
            -out {output file} \

Example:
    python mscoco_gt_boxes.py \
            -list mylist.txt \
            -out boxes.npy
"""

from data_api import MSCOCO
from utils import list_reader
from utils import logger
from utils import progress_bar
import argparse
import numpy
import os
import paths

log = logger.get()


def run(mscoco, image_list):
    """Run all images.

    Args:
        mscoco: MSCOCO API object.
        image_list: list, list of image IDs.
    """
    results = []
    not_found = []
    cat_rev_dict = mscoco.get_cat_list_reverse()
    
    log.info('Running through all images')
    for image_fname in progress_bar.get_list(image_list):
        image_id = get_image_id_from_path(image_fname)
        anns = mscoco.get_image_annotations(image_id)

        if anns is None:
            not_found.append(image_id)
            results.append(numpy.zeros((0, 5), dtype='int16'))
            continue

        num_ann = len(anns)
        result = numpy.zeros((num_ann, 5), dtype='int16')

        for i, ann in enumerate(anns):
            result[i, :4] = numpy.floor(ann['bbox']).astype('int16')
            result[i, 4] = cat_rev_dict[ann['category_id']]

        results.append(result)

    for image_id in not_found:
        log.error('Not found annotation for image {}'.format(image_id))
    log.error('Total {:d} image annotations not found.'.format(len(not_found)))

    results = numpy.array(results, dtype='object')
    
    return results


def save_boxes(output_file, boxes):
    """Saves bounding boxes.

    Args:
        output_file: string, path to the output numpy array.
        boxes: numpy.ndarray, bounding boxes.
    """

    numpy.save(output_file, boxes)

    pass


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Get groundtruth boxes in MS-COCO')
    parser.add_argument(
        '-list',
        dest='image_list',
        default=os.path.join(paths.GOBI_MREN_MSCOCO, 'image_list_train.txt'),
        help='image list text file')
    parser.add_argument(
        '-out',
        dest='output_file',
        default=os.path.join(paths.GOBI_MREN_MSCOCO, 'gt_boxes_train.npy'),
        help='output numpy file')
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
    log.info('Output file: {0}'.format(args.output_file))
    image_list = list_reader.read_file_list(args.image_list)
    mscoco = MSCOCO(base_dir=args.datadir, set_name=args.set)
    boxes = run(mscoco, image_list)
    save_boxes(args.output_file, boxes)
