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
import numpy as np

log = logger.get('../log/mscoco_gen_count_data')


def get_count(annotations, cat_dict):
    """Get count information for each object class.

    Returns:
        dict
        {
            'image_id': string.
            'count': K-dim vector.
        }
    """
    num_classes = len(cat_dict)
    counts = np.zeros((num_classes), dtype='int64')

    for ann in annotations:
        cat_id = cat_dict[ann['category']]
        counts[cat_id] += 1

    return {
        'image_id': annotations[0]['image_id'],
        'counts': counts
    }


def stat_count(counts_array):
    counts_sum = counts_array.sum(axis=0)
    counts_avg = counts_array.mean(axis=0)
    counts_dist = counts_sum / counts_sum.sum()
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
    cat_dict = mscoco.get_cat_dict()
    # num_classes = len(cat_dict)
    num_images = len(image_ids)

    counts_array = []
    counts_id = []
    log.info('Scanning dataset')
    for image_id in progress_bar.get(num_images, image_ids):
        annotations = mscoco.get_image_annotations(image_id)
        if len(annotations) > 0:
            counts_dict = get_count(annotations, cat_dict)
            counts_array.append(counts_dict['count'])
            counts_id.append(counts_dict['image_id'])

    counts_array = np.concatenate(counts_array, axis=0)
    stat_count(counts_array)
    pass
