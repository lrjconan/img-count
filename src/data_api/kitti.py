import sys
sys.path.insert(0, '../')
from utils import logger
from utils import progress_bar as pb
import cv2
import numpy as np
import os
import re
import h5py

log = logger.get()

"""
Image size are around 375 x 1240.
We may want to resize it first to
160 x 480 (32 padding)
128 x 448
64 x 224
32 x 112
16 x 56
8 x 28
4 x 14
"""


class KITTI(object):

    def __init__(self, folder, opt, split='train'):
        self.folder = folder
        self.opt = opt
        self.split = split
        self.dataset = None
        inp_height = self.opt['height']
        inp_width = self.opt['width']
        self.h5_fname = os.path.join(self.folder, '{}_{}x{}.h5'.format(
            self.split, inp_height, inp_width))

        if self.split == 'train' or self.split == 'valid':
            self.gt_folder = os.path.join(self.folder, 'gt')
        elif self.split == 'valid_man' or self.split == 'test_man':
            self.gt_folder = os.path.join(self.folder, 'gt_man')
        self.image_folder = os.path.join(self.folder, 'images')

        pass

    def get_dataset(self):
        """
        Recommended settings: 128 x 448.
        """
        if self.dataset is not None:
            return self.dataset

        cache = self.read_h5_data()
        if cache:
            self.dataset = cache
            return self.dataset

        inp_height = self.opt['height']
        inp_width = self.opt['width']
        num_ex = self.opt['num_examples'] if 'num_examples' in self.opt else -1
        timespan = self.opt['timespan'] if 'timespan' in self.opt else -1
        inp_shape = (inp_width, inp_height)

        ids_fname = os.path.join(self.folder, '{}.txt'.format(self.split))
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

        # Shuffle sequence.
        random = np.random.RandomState(2)
        shuffle = np.arange(len(img_ids))
        random.shuffle(shuffle)

        # Read images.
        log.info('Reading {} images'.format(num_ex))
        idx_map = np.zeros(len(img_ids), dtype='int')
        for idx in pb.get(num_ex):
            img_id = img_ids[shuffle[idx]]
            idx_map[idx] = int(img_id)
            fname = '{}.png'.format(img_id)
            img_fname = os.path.join(self.image_folder, fname)
            gt_fname = os.path.join(self.gt_folder, fname)

            img = cv2.imread(img_fname)
            img = cv2.resize(img, inp_shape, interpolation=cv2.INTER_NEAREST)

            gt = cv2.imread(gt_fname)
            gt = gt.astype('float')
            segm = self.get_separate_labels(gt)
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

        print idx_map
        self.dataset = {
            'input': inp,
            'label_segmentation': label_segm,
            'label_score': label_score,
            'index_map': idx_map
        }
        self.write_h5_data()

        return self.dataset

    def read_h5_data(self):
        """Read a dataset stored in H5."""
        if os.path.exists(self.h5_fname):
            log.info('Reading dataset from {}'.format(self.h5_fname))
            h5f = h5py.File(self.h5_fname, 'r')
            dataset = {}
            for key in h5f.keys():
                dataset[key] = h5f[key][:]
                pass

            return dataset

        else:
            return None

    def write_h5_data(self):
        log.info('Writing dataset to {}'.format(self.h5_fname))
        h5f = h5py.File(self.h5_fname, 'w')
        for key in dataset.iterkeys():
            h5f[key] = self.dataset[key]
            pass

    @staticmethod
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

    def get_labels(self, idx):
        num_ex = idx.shape[0]
        labels = []

        for ii in xrange(num_ex):
            img_fname = os.path.join(
                self.gt_folder, '{:06d}.png'.format(idx[ii]))
            img = cv2.imread(img_fname)
            labels.append(self.get_separate_labels(img))

        label_out = []
        for ii in xrange(num_ex):
            im_height = labels[ii][0].shape[0]
            im_width = labels[ii][0].shape[1]
            labels_out.append(np.zeros(
                [self.opt['timespan'], im_height, im_width], dtype='uint8'))
            for jj in xrange(len(labels[ii])):
                labels_out[ii, jj] = labels[ii][jj]

        return labels_out

# def get_foreground_dataset(folder, opt, split='train'):
#     h5_fname = os.path.join(folder, 'fg_' + split + '.h5')
#     cache = read_h5_data(h5_fname)
#     if cache:
#         return cache

#     ins_segm_data = get_dataset(folder, opt, split)
#     dataset = {
#         'input': ins_segm_data['input'],
#         'label': np.sum(ins_segm_data['label_segmentation'], axis=1)
#     }
#     write_h5_data(h5_fname, dataset)

#     return dataset


if __name__ == '__main__':
    folder = '/ais/gobi3/u/mren/data/kitti/object'
    # for split in ['valid_man', 'test_man', 'train', 'valid']:
    #     dataset = KITTI(
    #         folder,
    #         opt={
    #             'height': 128,
    #             'width': 448,
    #             'num_ex': -1,
    #             'timespan': 20
    #         },
    #         split=split).get_dataset()
    print KITTI(folder, opt={'height': 128,
                             'width': 448,
                             'num_ex': -1,
                             'timespan': 20
                             },
                split='valid_man').get_labels(np.array([1762, 5467]))
    pass
