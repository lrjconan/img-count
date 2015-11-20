"""
Generate counting questions and answers in MS-COCO dataset, using groundtruth 
annotation.

Ideal distribution:
(design a discrete distribution)

Usage: python mscoco_gen_count_data.py

"""

from data_api import Mscoco
from utils import logger
from utils import progress_bar
import argparse
import numpy as np

log = logger.get()

MAX_COUNT = 30

def get_count(annotations, cat_list_rev):
    """Get count information for each object class.

    Returns:
        dict
        {
            'image_id': string.
            'count': K-dim vector.
        }
    """
    num_classes = len(cat_list_rev)
    counts = np.zeros((num_classes), dtype='int64')
    for ann in annotations:
        cat_id = cat_list_rev[ann['category_id']]
        counts[cat_id] += 1

    counts_array = np.zeros((MAX_COUNT), dtype='int64')
    
    for i in xrange(num_classes):
        counts_array[counts[i]] += 1

    return {
        'image_id': annotations[0]['image_id'],
        'cat_counts': counts,
        'counts_array': counts_array
    }


def stat_count(counts_array):
    counts_sum = counts_array.sum(axis=0)
    counts_avg = counts_array.mean(axis=0)
    counts_dist = counts_sum / float(counts_sum.sum())
    log.info('Sum: {}'.format(counts_sum))
    log.info('Mean: {}'.format(counts_avg))
    log.info('Dist: {}'.format(counts_dist))

    pass


def apply_distribution(data, dist):
    """Apply a distribution on data. Reject too common labels.
    """
    pass


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Generate counting questions')
    parser.add_argument('-set', default='train', help='Train or valid set')
    parser.add_argument('-datadir', default='../data/mscoco',
                        help='Dataset directory')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    mscoco = Mscoco(base_dir=args.datadir, set_name=args.set)
    image_ids = mscoco.get_image_ids()
    cat_list_rev = mscoco.get_cat_list_reverse()
    # num_classes = len(cat_dict)
    num_images = len(image_ids)

    counts_array = []
    cat_counts = []
    counts_id = []
    err_image_ids = []
    empty_image_ids = []
    success_image_ids = []
    log.info('Scanning dataset')
    for image_id in progress_bar.get(num_images, iter(image_ids)):
        annotations = mscoco.get_image_annotations(image_id)
        if annotations is not None and len(annotations) > 0:
            counts_dict = get_count(annotations, cat_list_rev)
            counts_array.append(counts_dict['counts_array'])
            cat_counts.append(counts_dict['cat_counts'])
            counts_id.append(counts_dict['image_id'])
            success_image_ids.append(image_id)
        elif annotations is None:
            err_image_ids.append(image_id)
        elif len(annotations) == 0:
            empty_image_ids.append(image_id)

    log.info('Scan complete')
    log.info('Success: {:d} Error: {:d} Empty: {:d} Total: {:d}'.format(
        len(success_image_ids),
        len(err_image_ids),
        len(empty_image_ids),
        num_images))
    for image_id in empty_image_ids:
        log.error('Empty annotation in image: {}'.format(image_id))
    for image_id in err_image_ids:
        log.error('No annotation in image: {}'.format(image_id))

    counts_array = np.vstack(counts_array)
    log.info('Statistics')
    stat_count(counts_array)
    pass
