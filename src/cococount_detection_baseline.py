"""
Count by Detection.
Baseline for COCO-Count. Get sum of detections.

Usage:
    cococount_detection_baseline.py -cococount_datadir {COCO-Count folder} \
                                    -mscoco_datadir {MS-COCO folder} \
                                    -det {MSCOCO detection file pattern}

Example:
    cococount_detection_baseline.py -cococount_datadir "../data/cococount" \
                                    -mscoco_datadir "../data/mscoco" \
                                    -det "fast_rcnn_valid_conf_0.8_nms_0.3-*"
"""

from data_api import MSCOCO, COCOCount
from utils import logger
from utils import progress_bar
from utils import stats_tools
from utils.sharded_hdf5 import ShardedFile, ShardedFileReader
import argparse

log = logger.get()


def run(mscoco, cococount, detect_file_pattern):
    """Run detection baseline.

    Args:
        mscoco: MSCOCO API object.
        cococount: COCOCount API object.
        detect_file_pattern: File pattern of the detection results.
    """

    # Initialize files.
    detect_file = ShardedFile.from_pattern_read(detect_file_pattern)
    cococount_info_path = cococount.get_info_data_path()
    cococount_info_file = ShardedFile.from_pattern_read(cococount_info_path)

    # Declare counters.
    correct = 0
    total = 0
    pred = []
    labels = []
    with ShardedFileReader(cococount_info_file) as info_reader:
        with ShardedFileReader(detect_file) as detect_reader:
            for item in progress_bar.get_iter(info_reader):
                # Read image ID.
                image_id = item['image_id']
                image_path = mscoco.get_image_path(image_id)

                # Check detection exists.
                if image_path not in detect_reader:
                    # log.error(
                    #     'Detection not found for image: {}'.format(image_id))
                    continue

                # Read detection.
                dets = detect_reader[image_path]

                # Count same category.
                count = 0
                if dets['categories'].size == 1:
                    if dets['categories'] == item['category'] + 1:
                        count += 1
                else:
                    for i in xrange(dets['categories'].size):
                        # Detection category ID is 1-based.
                        if dets['categories'][i] == item['category'] + 1:
                            count += 1

                # Increment counters.
                if count > 0:
                    pred.append(count)
                    labels.append(item['number'])

                    if count == item['number']:
                        correct += 1

                    total += 1

    log.info('Number of examples: {:d}'.format(total))

    if total > 0:
        acc = correct / float(total)
        log.info('Accuracy: {:4f}'.format(acc))
        cf_mat = stats_tools.confusion_matrix(pred, labels)
        cf_mat_norm = stats_tools.confusion_matrix_norm(pred, labels)
        stats_tools.print_confusion_matrix(cf_mat, label_classes=range(1, 10))
        stats_tools.print_confusion_matrix(cf_mat_norm,
                                           label_classes=range(1, 10))

    return acc


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Generate counting questions')
    parser.add_argument(
        '-cococount_datadir', default='../data/cococount', help='Data directory')
    parser.add_argument(
        '-mscoco_datadir', default='../data/mscoco', help='Data directory')
    parser.add_argument('-det', help='MS-COCO detection file pattern')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    if not args.det:
        log.fatal('You need to provide detection file.')
    mscoco = MSCOCO(base_dir=args.mscoco_datadir, set_name='valid')
    cococount = COCOCount(base_dir=args.cococount_datadir, set_name='valid')
    run(mscoco, cococount, args.det)

    pass
