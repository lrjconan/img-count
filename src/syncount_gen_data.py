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

# Full image height.
kHeight = 224

# Full image width.
kWidth = 224

# Circle radius lower bound.
kRadiusLower = 5

# Circle radius upper bound.
kRadiusUpper = 20

# Number of examples.
kNumExamples = 100

# Maximum number of circles.
kMaxNumCircles = 10

# Random window size variance.
kSizeVar = 10

# Random window center variance.
kCenterVar = 5

# Resample window size (segmentation output unisize).
kOutputWindowSize = 128

# Ratio of negative and positive examples for segmentation data.
kNegPosRatio = 5

# Minimum window size of the original image (before upsampling).
kMinWindowSize = 20

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


def get_image_data(opt, raw_data):
    """Compile image and segmentation maps with object information.

    Args:
        opt
        raw_data
    Returns:
        results:
            images
            segmentations
            object_info
            questions
            answers
    """
    im_height = opt['height']
    im_width = opt['width']
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
        segms = np.zeros((num_circles, im_height, im_width), dtype='float32')
        # Information of the circles.
        obj_info = []

        # Compute images and segmentation maps (unoccluded).
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
            segms[jj] = segm
            obj_info.append(circle)

        # Apply occuclusion on segmentation map.
        if num_circles > 1:
            for jj in xrange(num_circles - 2, 0, -1):
                mask = np.logical_not(segms[jj + 1:, :, :].any(axis=0))
                log.info('Before sum: {}'.format(segms[jj].sum()), verbose=2)
                log.info('Top mask sum: {}'.format(mask.sum()), verbose=2)
                segms[jj] = np.logical_and(segms[jj], mask).astype('float32')
                log.info('After sum: {}'.format(segms[jj].sum()), verbose=2)

        # Aggregate results.
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


def get_segmentation_data(opt, image_data, seed=2):
    """
    Get dataset for training segmentation.
    Input images with the groundtruth segmentations.
    Output sliding windows of the image, and groudtruth to be the centering
    object. Random gittering and scaling will be applied. Resize image input
    to 128 x 128. Positive example : negative example = 1 : 5. Random sample
    sliding windows accross the image to generate negative examples.

    Args:
        opt: options.
        image_data: dataset generated from get_image_data.
    Returns:
        results: dictionary
            input_data
            label_segmentation
            label_objectness
    """
    random = np.random.RandomState(seed)

    # Constants
    min_window_size = opt['min_window_size']
    neg_pos_ratio = opt['neg_pos_ratio']
    radius_upper = opt['radius_upper']
    outsize = opt['output_window_size']
    output_window_size = (outsize, outsize)
    images = image_data['images']
    segmentations = image_data['segmentations']
    full_width = images[0].shape[1]
    full_height = images[0].shape[0]

    # Count total number of examples.
    num_ex = len(image_data['images'])
    num_ex_final = 0
    for ii in xrange(num_ex):
        num_ex_final += len(image_data['object_info'][ii])
    num_ex_final *= (1 + neg_pos_ratio)

    # Initialize arrays.
    input_data = np.zeros((num_ex_final, outsize, outsize, 3), dtype='float32')
    label_segmentation = np.zeros((num_ex_final, outsize, outsize),
                                  dtype='float32')
    label_objectness = np.zeros(num_ex_final, dtype='float32')

    log.info('Preparing segmentation data, {} examples'.format(num_ex))

    idx = 0
    for ii in progress_bar.get(num_ex):
        obj_info = image_data['object_info'][ii]
        for jj, obj in enumerate(obj_info):
            center = obj['center']
            radius = obj['radius']

            # Gittering on center.
            dx = int(np.ceil(random.normal(0, kCenterVar)))
            dy = int(np.ceil(random.normal(0, kCenterVar)))

            # Random scaling, window size (square shape).
            size = max(
                4 * radius + int(np.ceil(random.normal(0, kSizeVar))),
                min_window_size)

            # Final enter of the sliding window.
            xx = max(center[0] + dx, size)
            yy = max(center[1] + dy, size)
            log.info('dx: {}, dy: {}, xx: {}, yy:{}'.format(dx, dy, xx, yy),
                     verbose=2)
            log.info('Size: {}'.format(size),
                     verbose=2)
            log.info('Sliding window: {} {} {} {}'.format(
                     yy - size / 2, yy + size / 2,
                     xx - size / 2, xx + size / 2),
                     verbose=2)

            # Crop image and its segmentation.
            crop_imag = images[ii][yy - size / 2: yy + size / 2,
                                   xx - size / 2: xx + size / 2]
            crop_segm = segmentations[ii][jj, yy - size / 2: yy + size / 2,
                                          xx - size / 2: xx + size / 2]
            log.info('Cropped image size: {}'.format(crop_imag.shape),
                     verbose=2)
            log.info('Resized image size: {}'.format(output_window_size),
                     verbose=2)

            # Resample the image and segmentation to be uni-size.
            resize_imag = cv2.resize(crop_imag, output_window_size)
            resize_segm = cv2.resize(crop_segm, output_window_size)
            input_data[idx] = resize_imag
            label_segmentation[idx] = resize_segm
            label_objectness[idx] = 1
            idx += 1

            # Add negative examples with random sliding windows.
            for kk in xrange(neg_pos_ratio):
                keep_sample = True
                while keep_sample:
                    size = random.uniform(min_window_size, 4 * radius_upper)
                    x = np.ceil(random.uniform(0, full_width - size))
                    y = np.ceil(random.uniform(0, full_height - size))
                    center_x = x + size / 2
                    center_y = y + size / 2
                    # Check if the the center is not too close to any circle.
                    found = False
                    for ll, obj2 in enumerate(obj_info):
                        center = obj2['center']
                        radius = obj2['radius']
                        if (center_x - center[0]) ** 2 + \
                                (center_y - center[1]) ** 2 < 200:
                            found = True
                            break
                    if not found:
                        keep_sample = False

                crop_imag = images[ii][y: y + size, x: x + size]
                resize_imag = cv2.resize(crop_imag, output_window_size)
                resize_segm = np.zeros(output_window_size, dtype='float32')
                input_data[idx] = resize_imag
                label_segmentation[idx] = resize_segm
                label_objectness[idx] = 0
                idx += 1

    results = {
        'input': input_data,
        'label_segmentation': label_segmentation,
        'label_objectness': label_objectness
    }

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
    parser.add_argument('-neg_pos_ratio', default=kNegPosRatio, type=int,
                        help='Ratio between negative and positive examples')
    parser.add_argument('-min_window_size', default=kMinWindowSize, type=int,
                        help='Minimum window size to crop')
    parser.add_argument('-output_window_size', default=kOutputWindowSize,
                        type=int, help='Output segmentation window size')
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
        'max_num_circles': args.max_num_circles,
        'neg_pos_ratio': args.neg_pos_ratio,
        'min_window_size': args.min_window_size,
        'output_window_size': args.output_window_size
    }
    raw_data = get_raw_data(opt)
    image_data = get_image_data(opt, raw_data)
    log.info('Images: {}'.format(image_data['images'].shape))
    log.info('Segmentations: {}'.format(len(image_data['segmentations'])))
    log.info('Segmentation 1: {}'.format(image_data['segmentations'][0].shape))
    log.info('Questions: {}'.format(image_data['questions'].shape))
    log.info('Answers: {}'.format(image_data['answers'].shape))

    # for ii in xrange(opt['num_examples']):
    #     cv2.imshow('image', image_data['images'][ii])
    #     for jj in xrange(image_data['segmentations'][ii].shape[0]):
    #         cv2.imshow('segm{}'.format(jj), image_data['segmentations'][ii][jj])
    #     cv2.waitKey()

    segm_data = get_segmentation_data(opt, image_data)
    log.info('Segmentation examples: {}'.format(len(segm_data['input'])))
    log.info('Segmentation input: {}'.format(segm_data['input'].shape))
    log.info('Segmentation label: {}'.format(
        segm_data['label_segmentation'].shape))

    # for ii in xrange(10):
    #     cv2.imshow('input', segm_data['input'][ii])
    #     cv2.imshow('label', segm_data['label_segmentation'][ii])
    #     cv2.waitKey()
