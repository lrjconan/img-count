import sys
sys.path.insert(0, '../')
from utils import logger
from utils import progress_bar as pb
import cv2
import numpy as np
import os
import re

label_regex = re.compile('plant(?P<imgid>[0-9]{3})_label.png')
image_regex = re.compile('plant(?P<imgid>[0-9]{3})_rgb.png')
log = logger.get()


def get_dataset(folder, opt):
    """
    Get CVPPP dataset.
    Args:
        folder: Folder in which the dataset is stored.
        opt:
            height: resize the input image.
            width: resize the input image.

    """
    inp_height = opt['height']
    inp_width = opt['width']
    padding = opt['padding']
    inp_shape = (inp_height, inp_width)
    full_height = inp_height + 2 * padding
    full_width = inp_width + 2 * padding
    full_shape = (full_height, full_width)
    log.info('Reading images from {}'.format(folder))
    file_list = os.listdir(folder)
    image_dict = {}
    label_dict = {}
    max_num_obj = 0
    image_height = -1
    image_width = -1
    for fname in pb.get_iter(file_list):
        match = label_regex.search(fname)
        is_label = True
        if match:
            imgid = int(match.group('imgid'))
        else:
            match = image_regex.search(fname)
            imgid = int(match.group('imgid'))
            is_label = False

        img = cv2.imread(os.path.join(folder, fname))
        img = cv2.resize(img, inp_shape, interpolation=cv2.INTER_NEAREST)
        if is_label:
            # List of list of segmentation binary images.
            label_dict[imgid] = get_separate_labels(img)
            max_num_obj = max(max_num_obj, len(label_dict[imgid]))
        else:
            # List of images.
            image_dict[imgid] = img
            if image_height == -1:
                image_height = img.shape[0]
                image_width = img.shape[1]

    # Include one more
    max_num_obj += 1

    num_ex = len(image_dict)
    inp = np.zeros([num_ex, image_height, image_width, 3], dtype='uint8')
    label_segm = np.zeros(
        [num_ex, max_num_obj, image_height, image_width], dtype='uint8')
    label_score = np.zeros([num_ex, max_num_obj], dtype='uint8')
    log.info('Number of examples: {}'.format(num_ex))
    log.info('Input height: {} width: {}'.format(
        image_height, image_width))
    log.info('Input shape: {} label shape: {} {}'.format(
        inp.shape, label_segm.shape, label_score.shape))
    log.info('Assemble inputs')
    for ii, imgid in enumerate(image_dict.iterkeys()):
        if imgid not in label_dict:
            raise Exception('Image ID not match: {}'.format(imgid))
        inp[ii] = image_dict[imgid]
        num_obj = len(label_dict[imgid])
        for jj in xrange(num_obj):
            label_segm[ii, jj] = label_dict[imgid][jj]
        label_score[ii, :num_obj] = 1

    # Apply padding
    if padding > 0:
        inp_full = np.zeros([num_ex, full_height, full_width, 3], dtype='uint8')
        inp_full[:, padding: inp_height + padding, padding: inp_width + padding, :] = inp
        label_segm_full = np.zeros([num_ex, max_num_obj, full_height, full_width], dtype='uint8')
        label_segm_full[:, :, padding: inp_height + padding, padding: inp_width + padding] = label_segm
    else:
        inp_full = inp
        label_segm_full = label_segm

    return {
        'input': inp_full,
        'label_segmentation': label_segm_full,
        'label_score': label_score
    }


def get_separate_labels(label_img):
    # 64-bit encoding
    l64 = label_img.astype('uint64')
    # Single channel mapping
    l64i = ((l64[:, :, 0] << 16) + (l64[:, :, 1] << 8) + l64[:, :, 2])
    colors = np.unique(l64i)
    segmentations = []
    for c in colors:
        if c != 0:
            segmentation = (l64i == c).astype('uint8')
            # cv2.imshow('{}'.format(c), segmentation * 255)
            segmentations.append(segmentation)
    # cv2.waitKey()

    return segmentations

if __name__ == '__main__':
    get_dataset('/home/mren/data/LSCData/A1', {'height': 224, 'width': 224})
