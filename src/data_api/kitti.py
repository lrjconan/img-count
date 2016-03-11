import sys
sys.path.insert(0, '../')
from utils import logger
from utils import progress_bar as pb
import cv2
import numpy as np
import os
import re

log = logger.get()

def get_dataset(folder, split='train', opt=None):
    image_folder = os.path.join(folder, 'images')
    gt_folder = os.path.join(folder, 'gt')
    ids_fname = os.path.join(folder, '{}.txt'.format(split))
    with open(ids_fname) as f_ids:
        for ii in f_ids:
            log.info(ii)
            img_id = ii.strip('\n')
            fname = '{}.png'.format(img_id)
            img_fname = os.path.join(image_folder, fname)
            gt_fname = os.path.join(gt_folder, fname)
            log.info(img_fname)
            log.info(gt_fname)
            img = cv2.imread(img_fname)
            gt = cv2.imread(gt_fname)
            gt = gt.astype('float')
            # cv2.imshow('Image', img)
            segm = get_separate_labels(gt)
            log.info('Image: {}'.format(img.shape))
            log.info('GT: {}'.format(gt.shape))
            log.info('Number of segmenetations: {}'.format(len(segm)))
            for jj, ss in enumerate(segm):
                log.info('Segmentations {}: {}'.format(jj, ss.shape))
            # cv2.waitKey()


def get_separate_labels(label_img):
    # 64-bit encoding
    l64 = label_img.astype('uint64')
    # Single channel mapping
    l64i = ((l64[:, :, 0] << 16) + (l64[:, :, 1] << 8) + l64[:, :, 2])
    colors = np.unique(l64i)
    segmentations = []
    log.info('Number of unique colors: {}'.format(len(colors)))
    for c in colors:
        if c != 0:
            segmentation = (l64i == c).astype('uint8')
            # cv2.imshow('{}'.format(c), segmentation * 255)
            segmentations.append(segmentation)
    # cv2.waitKey()

    return segmentations


if __name__ == '__main__':
    folder = '/ais/gobi3/u/mren/data/kitti'

    # get_dataset('/home/mren/data/kitti/Images', '/home/mren/data/kitti/GTs',
    #             {'height': 224, 'width': 224})

    
    get_dataset(folder)
