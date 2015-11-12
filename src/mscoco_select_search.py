from __future__ import print_function
import argparse
import cv2
import logger
import numpy
import os
import progress_bar
import selective_search
import sys

gobi = '/ais/gobi3/u/mren/data/mscoco'
log_path = os.environ.get('LOG_OUTPUT', '../logs/mscoco_select_search')
log = logger.get(log_path)


def read_image_list(image_list_file):
    """
    Reads image file names in a plain text file.

    Args:
        image_list_file: string, path to line delimited image id file.
    Returns:
        image_list: list, list of image file names in string.
    Raises:
        Exception: image file name does not exists.
    """

    log.info('Checking all images')
    image_list = list(open(image_list_file))
    N = len(image_list)
    pb = progress_bar.get(N)
    for i in xrange(N):
        image_list[i] = image_list[i].strip()
        if not os.path.exists(image_list[i]):
            raise Exception('Image not found: {0}'.format(image_list[i]))
        pb.increment()
    return image_list


def patch(original_boxes, error_image_list, full_image_list, batch_size=10):
    """
    Pathches the search boxes for some of the images

    Args:
        original_boxes: numpy.ndarray, boxes to be patched.
        error_image_list: list, list of image filenames to be patched.
        full_image_list: list, full image list decribing each row of the array.
        batch_size: number, number of images to run in one batch.
    Returns:
        original_boxes: numpy.ndarray, updated search boxes.
    """
    boxes, processed, error = run_selective_search(
        error_image_list, batch_size=batch_size)
    image_list_dict = {}
    for idx, image in enumerate(full_image_list):
        image_list_dict[image] = idx
    for idx, image in enumerate(error_image_list):
        orig_idx = image_list_dict[image]
        original_boxes[orig_idx] = boxes[idx]
    return original_boxes


def run_selective_search(image_list, batch_size=10):
    """
    Run selective search on the list of images.
    Args:
        image_list: list, list of image file names.
    Returns:
        boxes: numpy.ndarray, list of box proposals.
    Notes:
        If error occurred in certain images, then zero entry will be filled in.
    """
    N = len(image_list)
    num_batch = int(numpy.ceil(N / float(batch_size)))
    log.info('Number of images: {0:d}'.format(N))
    log.info('Batch size: {0:d}'.format(batch_size))
    log.info('Number of batches: {0:d}'.format(num_batch))
    boxes = []
    processed_images = []
    error_images = []

    # Iterates through all images.
    for i in xrange(num_batch):
        a = i * batch_size
        b = min(N, (i + 1) * batch_size)
        image_list_batch = image_list[a: b]
        error_occurred = False
        try:
            boxes_batch = selective_search.get_windows(image_list_batch)
        except Exception as e:
            log.error('Error occurred')
            log.error(type(e))
            log.error(e)
            error_occurred = True
            for img in image_list_batch:
                log.error('Did not process {0}:'.format(img))
            error_images.extend(image_list_batch)
        if not error_occurred:
            boxes.extend(boxes_batch)
            processed_images.extend(image_list_batch)
            for img in image_list_batch:
                log.info('Processed {0}'.format(img))
        else:
            boxes.extend([0] * len(image_list_batch))
        log.info('Ran through {0}'.format(b))

    # Finally log all errors.
    for img in error_images:
        log.error('Did not process {0}'.format(img))

    # Pack into numpy format.
    boxes = numpy.array(boxes, dtype='object'))
    return boxes, processed_images, error_images


def save_boxes(output_file, boxes):
    """
    Saves computed search boxes.

    Args:
        output_file: string, path to the output numpy array.
    """

    numpy.save(output_file, boxes)
    pass


def load_boxes(output_file):
    """
    Loads computed search boxes.

    Args:
        output_file: string, path to the saved numpy array.
    """
    return numpy.load(output_file)

if __name__ == '__main__':
    """
    Runs selective search on a list of images.

    Usage: python mscoco_select_search.py \
            -list {image list} \
            -out {output file} \
            [-old {old file}] \
            [-patch {patch list}] \
            [-batch {batch size}]
    """
    parser=argparse.ArgumentParser(
        description = 'Compute selective search boxes on MS-COCO')
    parser.add_argument(
        '-list',
        dest = 'image_list',
        default = os.path.join(gobi, 'image_list_train.txt'),
        help = 'image list text file')
    parser.add_argument(
        '-out',
        dest = 'output_file',
        default = os.path.join(gobi, 'select_search_train.npy'),
        help = 'output numpy file')
    parser.add_argument(
        '-old',
        dest = 'to_patch_file',
        default = None,
        help = 'file to be patched')
    parser.add_argument(
        '-patch',
        dest = 'patch_list',
        default = None,
        help = 'list of images that needs to be patched')
    parser.add_argument(
        '-batch',
        dest = 'batch_size',
        type = int,
        default = 10,
        help = 'batch size')
    args=parser.parse_args()
    log.info('Input list: {0}'.format(args.image_list))
    log.info('Output file: {0}'.format(args.output_file))
    image_list=read_image_list(args.image_list)
    if args.patch_list is not None:
        if args.to_patch_file is None:
            raise Exception('Need to specify the file to be patched')
        log.info('File to be patched: {0}'.format(args.to_patch_file))
        log.info('Patch list: {0}'.format(args.patch_list))
        patch_list=read_image_list(args.patch_list)
        boxes=load_boxes(args.to_patch_file)
        boxes=patch(boxes, patch_list, image_list,
                      batch_size = args.batch_size)
    else:
        boxes, processed, error=run_selective_search(
            image_list, batch_size=args.batch_size)
        save_boxes(args.output_file, boxes)
    dirname = os.path.dirname(args.output_file)
    timestr = log.get_time_str()
    processed_filename = os.path.join(dirname,
                            'select_search_processed_{0}.txt'.format(timestr))
    error_filename = os.path.join(dirname,
                            'select_search_error_{0}.txt'.format(timestr))
    with open(processed_filename, 'w') as f:
        f.write('\n'.join(processed))
    with open(error_filename, 'w') as f:
        f.write('\n'.join(error))
    pass
