import sys
sys.path.insert(0, '../')
from utils import logger
from utils import progress_bar as pb
import cv2
import numpy as np
import os
import re

log = logger.get()

"""
Image size are around 375 x 1240.
We may want to resize it first to 
168 x 480 (32 padding)
136 x 448
68 x 224
34 x 112
17 x 56
8 x 28
4 x 14
"""


def get_dataset(folder, opt, split='train'):
    """
    Recommended settings: 136 x 448.
    """
    inp_height = opt['height']
    inp_width = opt['width']
    num_ex = opt['num_examples'] if 'num_examples' in opt else -1
    timespan = opt['timespan'] if 'timespan' in opt else -1
    inp_shape = (inp_width, inp_height)
    image_folder = os.path.join(folder, 'images')
    gt_folder = os.path.join(folder, 'gt')
    ids_fname = os.path.join(folder, '{}.txt'.format(split))
    inp_list = []
    segm_list = []
    max_num_obj = 0
    img_ids = []

    log.info('Reading image IDs')
    with open(ids_fname) as f_ids:
        for ii in f_ids:
            img_ids.append(ii.strip('\n'))
   
    if num_ex == -1:
        num_ex = len(img_ids)

    # Read images.
    log.info('Reading {} images'.format(num_ex))
    for img_id in pb.get_iter(img_ids[: num_ex]):
        fname = '{}.png'.format(img_id)
        img_fname = os.path.join(image_folder, fname)
        gt_fname = os.path.join(gt_folder, fname)

        img = cv2.imread(img_fname)
        img = cv2.resize(img, inp_shape, interpolation=cv2.INTER_NEAREST)

        gt = cv2.imread(gt_fname)
        gt = gt.astype('float')
        segm = get_separate_labels(gt)
        max_num_obj = max(max_num_obj, len(segm))
        segm_reshape = []
        for jj, ss in enumerate(segm):
            segm_reshape.append(cv2.resize(
                ss, inp_shape, interpolation=cv2.INTER_NEAREST))

        inp_list.append(img)
        segm_list.append(segm_reshape)

    # Include one more
    max_num_obj += 1

    if timespan == -1:
        timespan = max_num_obj
    else:
        timespan = max(timespan, max_num_obj)

    log.info('Assemble images')
    num_ex = len(inp_list)
    inp = np.zeros([num_ex, inp_height, inp_width, 3], dtype='uint8')
    label_segm = np.zeros(
        [num_ex, timespan, inp_height, inp_width], dtype='uint8')
    label_score = np.zeros([num_ex, timespan], dtype='uint8')
    log.info('Number of examples: {}'.format(num_ex))
    log.info('Input height: {} width: {}'.format(inp_height, inp_width))
    log.info('Input shape: {} label shape: {} {}'.format(
        inp.shape, label_segm.shape, label_score.shape))

    for ii, data in enumerate(zip(inp_list, segm_list)):
        img = data[0]
        segm = data[1]
        inp[ii] = img
        num_obj = len(segm)
        for jj in xrange(num_obj):
            label_segm[ii, jj] = segm[jj]
        label_score[ii, :num_obj] = 1

    cv2.waitKey()
    return {
        'input': inp,
        'label_segmentation': label_segm,
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
    folder = '/ais/gobi3/u/mren/data/kitti'
    get_dataset(folder, {'height': 136, 'width': 448, 'num_ex': 100})
