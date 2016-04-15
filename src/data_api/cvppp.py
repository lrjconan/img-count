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


class CVPPP(object):

    def __init__(self, folder, opt, split=None):
        self.folder = folder
        self.opt = opt
        self.split = split
        self.dataset = None
        pass

    def get_dataset(self):
        """
        Get CVPPP dataset.
        Args:
            folder: Folder in which the dataset is stored.
            opt:
                height: resize the input image.
                width: resize the input image.
            split: 'train' or 'valid' or None
        """

        if self.dataset is not None:
            return self.dataset

        inp_height = self.opt['height']
        inp_width = self.opt['width']
        inp_shape = (inp_width, inp_height)
        log.info('Reading images from {}'.format(self.folder))
        file_list = os.listdir(self.folder)
        image_dict = {}
        label_dict = {}
        max_num_obj = 0

        split_ids = None
        if self.split is not None:
            id_fname = os.path.join(self.folder, '{}.txt'.format(self.split))
            if not os.path.exists(id_fname):
                self.write_split()
            with open(id_fname) as f:
                split_ids = set([int(ll.strip('\n')) for ll in f.readlines()])

        for fname in pb.get_iter(file_list):
            match = label_regex.search(fname)
            is_label = True
            matched = False
            if match:
                imgid = int(match.group('imgid'))
                matched = True
            else:
                match = image_regex.search(fname)
                if match:
                    imgid = int(match.group('imgid'))
                    is_label = False
                    matched = True

            if not matched:
                continue

            if matched:
                if split_ids is not None and imgid not in split_ids:
                    continue

            img = cv2.imread(os.path.join(self.folder, fname))
            img = cv2.resize(img, inp_shape, interpolation=cv2.INTER_NEAREST)
            if is_label:
                # List of list of segmentation binary images.
                label_dict[imgid] = self.get_separate_labels(img)
                max_num_obj = max(max_num_obj, len(label_dict[imgid]))
            else:
                # List of images.
                image_dict[imgid] = img

        # Include one more
        max_num_obj += 1

        self.max_num_obj = max_num_obj

        num_ex = len(image_dict)
        inp = np.zeros([num_ex, inp_height, inp_width, 3], dtype='uint8')
        label_segm = np.zeros(
            [num_ex, max_num_obj, inp_height, inp_width], dtype='uint8')
        label_score = np.zeros([num_ex, max_num_obj], dtype='uint8')
        log.info('Number of examples: {}'.format(num_ex))
        log.info('Input height: {} width: {}'.format(inp_height, image_width))
        log.info('Input shape: {} label shape: {} {}'.format(
            inp.shape, label_segm.shape, label_score.shape))
        log.info('Assemble inputs')

        idx_map = np.array(image_dict.keys())
        for ii, imgid in enumerate(image_dict.iterkeys()):
            inp[ii] = image_dict[imgid]

            if imgid in label_dict:
                # For test set, it is not available.
                num_obj = len(label_dict[imgid])
                for jj in xrange(num_obj):
                    label_segm[ii, jj] = label_dict[imgid][jj]
                label_score[ii, :num_obj] = 1

        # Shuffle the indices.
        random = np.random.RandomState(2)
        idx = np.arange(idx_map.size)
        random.shuffle(idx)
        inp = inp[idx]
        label_segm = label_segm[idx]
        label_score = label_score[idx]
        idx_map = idx_map[idx]

        self.dataset = {
            'input': inp,
            'label_segmentation': label_segm,
            'label_score': label_score,
            'index_map': idx_map
        }

        return self.dataset

    def write_split(self):
        random = np.random.RandomState(2)
        log.info('Reading images from {}'.format(self.folder))
        file_list = os.listdir(self.folder)

        image_id_dict = {}
        for fname in pb.get_iter(file_list):
            match = label_regex.search(fname)
            if match:
                imgid = int(match.group('imgid'))
                image_id_dict[imgid] = True

        image_ids = np.array(image_id_dict.keys())
        num_train = np.ceil(image_ids.size * 0.8)  # 103 for 128
        idx = np.arange(len(image_ids))
        random.shuffle(idx)
        train_ids = image_ids[idx[: num_train]]
        valid_ids = image_ids[idx[num_train:]]
        log.info('Number of images: {}'.format(len(idx)))

        log.info('Train indices: {}'.format(idx[: num_train]))
        log.info('Valid indices: {}'.format(idx[num_train:]))
        log.info('Train image ids: {}'.format(train_ids))
        log.info('Valid image ids: {}'.format(valid_ids))

        with open(os.path.join(self.folder, 'train.txt'), 'w') as f:
            for ii in train_ids:
                f.write('{}\n'.format(ii))

        with open(os.path.join(self.folder, 'valid.txt'), 'w') as f:
            for ii in valid_ids:
                f.write('{}\n'.format(ii))

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

    def get_label(idx):
        im_height = -1
        im_width = -1
        labels = []

        for ii in xrange(idx.shape[0]):
            img_fname = os.path.join(
                self.folder, 'plant{:03d}_label.png'.format(idx[ii]))
            img = cv2.imread(img_fname)
            labels.append(self.get_separate_labels(img_fname))
            if im_height == -1:
                im_height = img.shape[0]
                im_width = img.shape[1]

        labels_out = np.zeros(
            [idx.shape[0], self.max_num_obj, im_height, im_width], dtype='uint8')
        for ii in xrange(idx.shape[0]):
            for jj in xrange(len(labels[ii])):
                labels_out[ii, jj] = labels[ii][jj]

        return labels_out


if __name__ == '__main__':
    if os.path.exists('/u/mren'):
        train_folder = '/ais/gobi3/u/mren/data/lsc'
        test_folder = '/ais/gobi3/u/mren/data/lsc_test'
    else:
        train_folder = '/home/mren/data/LSCData'
        test_folder = '/home/mren/data/LSCDataTest'

    for subset in ['A1', 'A2', 'A3']:
        CVPPP(os.path.join(train_folder, subset), None).write_split()

    d = CVPPP(os.path.join(train_folder, 'A1'),
              {'height': 224, 'width': 224},
              split='train').get_dataset()
    print d['input'].shape
    print d['label_segmentation'].shape
    d = CVPPP(os.path.join(train_folder, 'A1'),
              {'height': 224, 'width': 224},
              split='valid').get_dataset()
    print d['input'].shape
    print d['label_segmentation'].shape
    d = CVPPP(os.path.join(test_folder, 'A1'),
              {'height': 224, 'width': 224},
              split=None).get_dataset()
    print d['input'].shape
    print d['label_segmentation'].shape
