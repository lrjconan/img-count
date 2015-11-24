"""
Runs selective search on a list of images.

Usage:
    python mscoco_select_search.py -image_list {image list} \
                                   -out {output file}

Example:
    python mscoco_select_search.py -image_list mylist.txt \
                                   -out boxes
"""

from __future__ import print_function
from utils import list_reader
from utils import logger
from utils.sharded_hdf5 import ShardedFile, ShardedFileWriter
import argparse
import cv2
import numpy
import os
import paths
import selective_search
import sys

log = logger.get()


def patch(original_boxes, error_image_list, full_image_list, batch_size=10):
    """Pathches the search boxes for some of the images

    Args:
        original_boxes: numpy.ndarray, boxes to be patched.
        error_image_list: list, list of image filenames to be patched.
        full_image_list: list, full image list decribing each row of the array.
        batch_size: number, number of images to run in one batch.
    Returns:
        original_boxes: numpy.ndarray, updated search boxes.
    """
    raise Exception('Not implemented')
    # boxes, processed, error = run_selective_search(
    #     error_image_list, batch_size=batch_size)
    # image_list_dict = {}
    # for idx, image in enumerate(full_image_list):
    #     image_list_dict[image] = idx
    # for idx, image in enumerate(error_image_list):
    #     orig_idx = image_list_dict[image]
    #     original_boxes[orig_idx] = boxes[idx]

    # return original_boxes


def run_selective_search(image_list, output_fname, num_shards=1):
    """Run selective search on the list of images.

    Args:
        image_list: list, list of image file names.
    Returns:
        boxes: numpy.ndarray, list of box proposals.
    Notes:
        If error occurred in certain images, then zero entry will be filled in.
    """
    num_images = len(image_list)
    processed_images = []
    error_images = []
    fout = ShardedFile(output_fname, num_shards=num_shards)

    with ShardedFileWriter(fout, num_objects=num_images) as writer:
        for i in writer:
            image_fname = image_list[i]
            error_occurred = False

            try:
                boxes = selective_search.get_windows([image_fname])
                boxes = boxes[0]
            except Exception as e:
                log.error('Error occurred while running selective search')
                log.error(type(e))
                log.error(e)
                log.log_exception(e)
                error_occurred = True
                log.error('Did not process {0}:'.format(image_fname))
                error_images.append(image_fname)
                boxes = numpy.zeros(4, dtype='int16')

            if not error_occurred:
                processed_images.append(image_fname)
                log.info('Processed {:d}/{:d}: {}'.format(i,
                                                          num_images, 
                                                          image_fname))

            data = {'boxes': boxes}
            writer.write(data, key=image_fname)

    # Finally log all errors.
    for img in error_images:
        log.error('Did not process {0}'.format(img))

    return processed_images, error_images


def save_boxes(output_file, boxes):
    """Saves computed search boxes.

    Args:
        output_file: string, path to the output numpy array.
        boxes: numpy.ndarray, computed search boxes.
    """
    raise Exception('Not implemented')
    # numpy.save(output_file, boxes)

    pass


def load_boxes(output_file):
    """Loads computed search boxes.

    Args:
        output_file: string, path to the saved numpy array.
    """
    return numpy.load(output_file)


def parse_args():
    """Parse arguments."""

    default_input = os.path.join(
        paths.GOBI_MREN_MSCOCO, 'image_list_train.txt')
    default_output = os.path.join(
        paths.GOBI_MREN_MSCOCO, 'select_search_train')
    parser = argparse.ArgumentParser(
        description='Compute selective search boxes in MS-COCO')
    parser.add_argument('-image_list',
                        default=default_input,
                        help='Image list text file')
    parser.add_argument('-output',
                        default=default_output,
                        help='Output file name')
    parser.add_argument('-num_shards',
                        default=1,
                        type=int,
                        help='Total number of shards')
    # parser.add_argument(
    #     '-old',
    #     dest='to_patch_file',
    #     default=None,
    #     help='file to be patched')
    # parser.add_argument(
    #     '-patch',
    #     dest='patch_list',
    #     default=None,
    #     help='list of images that needs to be patched')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    log.info('Input list: {0}'.format(args.image_list))
    log.info('Output file: {0}'.format(args.output))
    log.info('Number of shards: {0}'.format(args.num_shards))
    image_list = list_reader.read_file_list(args.image_list, check=True)

    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # if args.patch_list is not None:
    #     if args.to_patch_file is None:
    #         raise Exception('Need to specify the file to be patched')
    #     log.info('File to be patched: {0}'.format(args.to_patch_file))
    #     log.info('Patch list: {0}'.format(args.patch_list))
    #     patch_list = list_reader.read_file_list(args.patch_list, check=True)
    #     boxes = load_boxes(args.to_patch_file)
    #     boxes = patch(boxes, patch_list, image_list,
    #                   batch_size=args.batch_size)
    # else:

    processed, error = run_selective_search(
        image_list, args.output, num_shards=args.num_shards)

    timestr = log.get_time_str()
    processed_fname_templ = 'select_search_processed_{0}.txt'
    processed_fname = os.path.join(
        dirname, processed_fname_templ.format(timestr))
    error_fname_templ = 'select_search_error_{0}.txt'
    error_fname = os.path.join(dirname, error_fname_templ.format(timestr))

    with open(processed_fname, 'w') as f:
        f.write('\n'.join(processed))
    with open(error_fname, 'w') as f:
        f.write('\n'.join(error))
