"""
Generate counting questions based on synthetic images.
Object categories are:
* Triangles (0)
* Circles (1)
* Squares (2)

For now, only single colour is present (i.e. green), but can be extended to 
multiple colours.
All objects has white border.
Background is black.

Output:
* Images: 224x224
* Groundtruth segmentations for each object: Mx224x224 (M objects)
* Question: int (class ID)
* Answer: int

Usage: python syncount_gen_data.py
"""

from utils import logger
from utils import progress_bar
import argparse
import cv2
import numpy as np

# Default constants
kHeight = 224
kWidth = 224
kRadiusLower = 5
kRadiusUpper = 20
kNumExamples = 100
kMaxNumCircles = 10

# Logger
log = logger.get()


def get_raw_data(opt, seed=2):
    """Generate raw data (dictionary).

    Args:
        opt: dictionary, options.
        seed:
    Returns:
        results:
    """

    # Read options.
    height = opt['height']
    width = opt['width']
    radius_upper = opt['radius_upper']
    radius_lower = opt['radius_lower']
    num_examples = opt['num_examples']
    max_num_circles = opt['max_num_circles']

    results = {}
    results['height'] = height
    results['width'] = width
    results['radius_upper'] = radius_upper
    results['radius_lower'] = radius_lower
    results['max_num_circles'] = max_num_circles
    examples = []
    random = np.random.RandomState(seed)

    log.info('Generating raw data')
    for ii in progress_bar.get(num_examples):
        num_circles = int(np.ceil(random.uniform(0, 1) * max_num_circles))
        ex = []
        for jj in xrange(num_circles):
            radius = int(np.ceil(random.uniform(radius_lower, radius_upper)))
            center_x = int(np.ceil(random.uniform(radius, width - radius)))
            center_y = int(np.ceil(random.uniform(radius, height - radius)))
            center = (center_x, center_y)
            ex.append({
                'center': center,
                'radius': radius,
                'type': 0  # Potentially useful for multiple object types.
            })
        examples.append(ex)

    results['examples'] = examples

    return results


def get_image_data(raw_data):
    im_height = raw_data['height']
    im_width = raw_data['width']
    examples = raw_data['examples']
    num_ex = len(examples)
    images = np.zeros((num_ex, im_height, im_width, 3), dtype='float32')
    segmentations = []
    object_info = []
    questions = np.zeros((num_ex), dtype='int16')
    answers = np.zeros((num_ex), dtype='int16')

    log.info('Compiling images, {} examples'.format(num_ex))
    for ii in progress_bar.get(num_ex):
        ex = examples[ii]
        img = np.zeros((im_height, im_width, 3))
        num_circles = len(ex)
        # Answer to the counting question.
        answers[ii] = num_circles
        # Segmentations of all circles (N, H, W) in one example, N is number
        # of circles.
        segms = []
        # Information of the circles.
        obj_info = []
        for jj, circle in enumerate(ex):
            radius = circle['radius']
            center = circle['center']
            segm = np.zeros((im_height, im_width), dtype='float32')

            # Draw a green circle with red border.
            # Note: OpenCV uses (B, G, R) order.
            cv2.circle(img,
                       center=center, radius=radius + 1,
                       color=(0, 0, 255), thickness=2)
            cv2.circle(img,
                       center=center, radius=radius,
                       color=(0, 255, 0), thickness=-1)
            cv2.circle(segm,
                       center=center, radius=radius,
                       color=(1, 0, 0), thickness=-1)
            segms.append(segm)
            obj_info.append(circle)

        segms = np.array(segms)
        segmentations.append(segms)
        object_info.append(obj_info)
        images[ii] = img

    results = {}
    results['images'] = images
    results['segmentations'] = segmentations
    results['object_info'] = object_info
    results['questions'] = questions
    results['answers'] = answers

    return results


def get_dataset(opt, num_train, num_valid):
    """
    Get dataset with train-test split.

    Args:
        opt:
        num_train:
        num_valid:
    Returns:
    """
    dataset = {}
    opt['num_examples'] = num_train
    raw_data = get_raw_data(opt, seed=2)
    image_data = get_image_data(raw_data)
    dataset['train'] = image_data

    opt['num_examples'] = num_valid
    raw_data = get_raw_data(opt, seed=3)
    image_data = get_image_data(raw_data)
    dataset['valid'] = image_data

    return dataset


def get_segmentation_training_data(image_data, seed=2):
    """
    Get dataset for training segmentation.
    Input images with the groundtruth segmentations.
    Output sliding windows of the image, and groudtruth to be the centering
    object. Random gittering and scaling will be applied. Resize image input
    to 128 x 128. Positive example : negative example = 1 : 5. Random sample
    sliding windows accross the image to generate negative examples.

    Args:
        image_data: dataset generated from get_image_data.
    Returns:
        results: dictionary
            input_data
            label_segmentation
            label_objectness
    """
    random = np.random.RandomState(seed)
    size_var = 10
    center_var = 10
    dst_image_size = (128, 128)
    neg_pos_ex_ratio = 5
    min_window_size = 20
    images = image_data['images']
    segmentations = image_data['segmentations']
    full_width = images[0].shape[1]
    full_height = images[0].shape[0]
    input_data = []
    label_segmentation = []
    label_objectness = []

    num_ex = len(image_data['images'])
    log.info('Preparing segmentation data, {} examples'.format(num_ex))
    for ii in progress_bar.get(num_ex):
        obj_info = image_data['object_info'][ii]
        for jj, obj in enumerate(obj_info):
            center = obj['center']
            radius = obj['radius']

            # Gittering on center.
            dx = int(np.ceil(random.normal(0, center_var)))
            dy = int(np.ceil(random.normal(0, center_var)))

            # Random scaling, window size (square shape).
            size = max(
                       2 * radius + int(np.ceil(random.normal(0, size_var))), 
                       min_window_size)

            # Final enter of the sliding window.
            xx = max(center[0] + dx, size)
            yy = max(center[1] + dy, size)
            # log.info('dx: {}, dy: {}, xx: {}, yy:{}'.format(dx, dy, xx, yy))
            # log.info('Size: {}'.format(size))
            # log.info('Sliding window: {} {} {} {}'.format(
            #          yy - size / 2, yy + size / 2,
            #          xx - size / 2, xx + size / 2))

            # Crop image and its segmentation.
            crop_imag = images[ii][yy - size / 2 : yy + size / 2,
                                   xx - size / 2 : xx + size / 2]
            crop_segm = segmentations[ii][jj, yy - size / 2 : yy + size / 2,
                                          xx - size / 2 : xx + size / 2]
            # log.info('Cropped image size: {}'.format(crop_imag.shape))
            # log.info('Resized image size: {}'.format(dst_image_size))

            # Resample the image and segmentation to be uni-size.
            resize_imag = cv2.resize(crop_imag, dst_image_size)
            resize_segm = cv2.resize(crop_segm, dst_image_size)
            input_data.append(resize_imag)
            log.info(resize_imag.shape)
            label_segmentation.append(resize_segm)
            label_objectness.append(1)

            # Add negative examples with random sliding windows.
            for kk in xrange(neg_pos_ex_ratio):
                # Randomly sample sliding windows as negative examples.
                x = np.ceil(random.uniform(0, full_width - dst_image_size[0]))
                y = np.ceil(random.uniform(0, full_height - dst_image_size[1]))
                crop_imag = images[ii][y : y + dst_image_size[1],
                                       x : x + dst_image_size[0]]
                crop_segm = np.zeros(dst_image_size, dtype='float32')
                input_data.append(crop_imag)
                label_segmentation.append(crop_segm)
                label_objectness.append(0)
    results = {
        'input': np.array(input_data, dtype='float32'),
        'label_segmentation': np.array(label_segmentation, dtype='float32'),
        'label_objectness': np.array(label_objectness, dtype='float32')
    }
    # results = {
    #     'input': input_data,
    #     'label_segmentation': label_segmentation,
    #     'label_objectness': label_objectness
    # }


    return results



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Generate counting questions on synthetic images')
    parser.add_argument('-height', default=kHeight, type=int,
                        help='Image height')
    parser.add_argument('-width', default=kWidth, type=int,
                        help='Image width')
    parser.add_argument('-radius_upper', default=kRadiusUpper, type=int,
                        help='Radius upper bound')
    parser.add_argument('-radius_lower', default=kRadiusLower, type=int,
                        help='Radius lower bound')
    parser.add_argument('-num', default=kNumExamples, type=int,
                        help='Number of examples')
    parser.add_argument('-max_num_circles', default=kMaxNumCircles, type=int,
                        help='Maximum number of circles')
    parser.add_argument('-output', default=None, help='Output file name')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    opt = {
        'height': args.height,
        'width': args.width,
        'radius_upper': args.radius_upper,
        'radius_lower': args.radius_lower,
        'num_examples': args.num,
        'max_num_circles': args.max_num_circles
    }
    raw_data = get_raw_data(opt)
    image_data = get_image_data(raw_data)
    log.info('Images: {}'.format(image_data['images'].shape))
    log.info('Segmentations: {}'.format(len(image_data['segmentations'])))
    log.info('Segmentation 1: {}'.format(image_data['segmentations'][0].shape))
    log.info('Questions: {}'.format(image_data['questions'].shape))
    log.info('Answers: {}'.format(image_data['answers'].shape))

    segm_data = get_segmentation_training_data(image_data)
    log.info('Segmentation examples: {}'.format(len(segm_data['input'])))
