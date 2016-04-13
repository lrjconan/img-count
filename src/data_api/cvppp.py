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


def get_dataset(folder, opt, split=None):
    """
    Get CVPPP dataset.
    Args:
        folder: Folder in which the dataset is stored.
        opt:
            height: resize the input image.
            width: resize the input image.
        split: 'train' or 'valid' or None

    """
    inp_height = opt['height']
    inp_width = opt['width']
    inp_shape = (inp_width, inp_height)
    log.info('Reading images from {}'.format(folder))
    file_list = os.listdir(folder)
    image_dict = {}
    label_dict = {}
    max_num_obj = 0
    image_height = -1
    image_width = -1

    split_ids = None
    if split is not None:
        with open(os.path.join(folder, '{}.txt'.format(split))) as f:
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

    return {
        'input': inp,
        'label_segmentation': label_segm,
        'label_score': label_score,
        'index_map': idx_map
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


def get_image(image_id):
    pass


def write_split(folder):
    random = np.random.RandomState(2)
    log.info('Reading images from {}'.format(folder))
    file_list = os.listdir(folder)

    image_id_dict = {}
    for fname in pb.get_iter(file_list):
        match = label_regex.search(fname)
        if match:
            imgid = int(match.group('imgid'))
            image_id_dict[imgid] = True


    image_ids = np.array(image_id_dict.keys())
    num_train = np.ceil(image_ids.size * 0.8) # 103 for 128
    idx = np.arange(len(image_ids))
    random.shuffle(idx)
    train_ids = image_ids[idx[: num_train]]
    valid_ids = image_ids[idx[num_train:]]
    log.info('Number of images: {}'.format(len(idx)))

    log.info('Train indices: {}'.format(idx[: num_train]))
    log.info('Valid indices: {}'.format(idx[num_train: ]))
    log.info('Train image ids: {}'.format(train_ids))
    log.info('Valid image ids: {}'.format(valid_ids))

    with open(os.path.join(folder, 'train.txt'), 'w') as f:
        for ii in train_ids:
            f.write('{}\n'.format(ii))

    with open(os.path.join(folder, 'valid.txt'), 'w') as f:
        for ii in valid_ids:
            f.write('{}\n'.format(ii))


if __name__ == '__main__':
    if os.path.exists('/u/mren'):
        train_folder = '/ais/gobi3/u/mren/data/lsc'
        test_folder = '/ais/gobi3/u/mren/data/lsc_test'
    else:
        train_folder = '/home/mren/data/LSCData'
        test_folder = '/home/mren/data/LSCDataTest'

    for subset in ['A1', 'A2', 'A3']:
        write_split(os.path.join(train_folder, subset))

    d = get_dataset(os.path.join(train_folder, 'A1'),
                    {'height': 224, 'width': 224},
                    split='train')
    print d['input'].shape
    print d['label_segmentation'].shape
    d = get_dataset(os.path.join(train_folder, 'A1'),
                    {'height': 224, 'width': 224},
                    split='valid')
    print d['input'].shape
    print d['label_segmentation'].shape
    d = get_dataset(os.path.join(test_folder, 'A1'),
                    {'height': 224, 'width': 224},
                    split=None)
    print d['input'].shape
    print d['label_segmentation'].shape
