"""
Generate counting questions based on synthetic images.
Object categories are:
* Circles (0)
* Triangles (1)
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
import matplotlib.pyplot as plt
import numpy as np

# Default constants

# Full image height.
kHeight = 224

# Full image width.
kWidth = 224

# Object radius lower bound.
kRadiusLower = 15

# Object radius upper bound.
kRadiusUpper = 25

# Object border thickness.
kBorderThickness = 2

# Number of examples.
kNumExamples = 100

# Maximum number of objects.
kMaxNumObjects = 10

# Number of object types, currently support up to three types (circles,
# triangles, and squares).
kNumObjectTypes = 3

# Random window size variance.
kSizeVar = 10

# Random window center variance.
kCenterVar = 0.01

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
    num_types = opt['num_object_types']
    max_num_objects = opt['max_num_objects']

    results = {}
    examples = []
    random = np.random.RandomState(seed)

    log.info('Generating raw data')
    for ii in progress_bar.get(num_examples):
        num_obj = int(np.ceil(random.uniform(0, 1) * max_num_objects))
        ex = []
        for jj in xrange(num_obj):
            radius = int(np.ceil(random.uniform(radius_lower, radius_upper)))
            center_x = int(np.ceil(random.uniform(radius, width - radius)))
            center_y = int(np.ceil(random.uniform(radius, height - radius)))
            center = (center_x, center_y)
            obj_type = int(np.floor(random.uniform(0, num_types - 1e-5)))
            ex.append({
                'center': center,
                'radius': radius,
                'type': obj_type
            })
        examples.append(ex)

    results['examples'] = examples

    return results


def _draw_circle(img, center, radius, fill, border=None, thickness=None):
    """Draw a circle given center and radius."""
    cv2.circle(img, center=center, radius=radius, color=fill, thickness=-1)

    if border:
        cv2.circle(img, center=center, radius=radius,
                   color=border, thickness=thickness)

    pass


def _draw_square(img, center, radius, fill, border=None, thickness=None):
    """Draw a square given center and radius."""
    top_left = (int(np.floor(center[0] - radius / np.sqrt(2))),
                int(np.floor(center[1] - radius / np.sqrt(2))))
    bottom_right = (int(np.floor(center[0] + radius / np.sqrt(2))),
                    int(np.floor(center[1] + radius / np.sqrt(2))))
    cv2.rectangle(img, pt1=top_left, pt2=bottom_right,
                  color=fill, thickness=-1)
    if border:
        cv2.rectangle(img, pt1=top_left, pt2=bottom_right,
                      color=border, thickness=thickness)

    pass


def _draw_triangle(img, center, radius, fill, border=None, thickness=None):
    """Draw a equilateral triangle given center and radius."""
    p1 = (int(center[0]), int(np.floor(center[1] - radius)))
    p2 = (int(np.floor(center[0] - radius * np.sin(60.0 * np.pi / 180.0))),
          int(np.floor(center[1] + radius * np.sin(30.0 * np.pi / 180.0))))
    p3 = (int(np.floor(center[0] + radius * np.sin(60.0 * np.pi / 180.0))),
          int(np.floor(center[1] + radius * np.sin(30.0 * np.pi / 180.0))))
    pts = np.array([p1, p2, p3])
    # log.info(pts)
    cv2.fillConvexPoly(img, points=pts, color=fill)

    if border:
        cv2.line(img, pt1=p1, pt2=p2, color=border, thickness=thickness)
        cv2.line(img, pt1=p2, pt2=p3, color=border, thickness=thickness)
        cv2.line(img, pt1=p3, pt2=p1, color=border, thickness=thickness)

    pass


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
    thickness = opt['border_thickness']

    # Draw a green objects with red border.
    # Note: OpenCV uses (B, G, R) order.
    fill_color = (0, 255, 0)
    border_color = (0, 0, 255)
    segm_color = (1, 0, 0)

    examples = raw_data['examples']
    num_ex = len(examples)
    images = np.zeros((num_ex, im_height, im_width, 3), dtype='uint8')
    segmentations = []
    object_info = []
    questions = np.zeros((num_ex), dtype='int16')
    answers = np.zeros((num_ex), dtype='int16')

    log.info('Compiling images, {} examples'.format(num_ex))
    for ii in progress_bar.get(num_ex):
        ex = examples[ii]
        img = np.zeros((im_height, im_width, 3))
        num_obj = len(ex)
        # Answer to the counting question.
        answers[ii] = num_obj
        # Segmentations of all objects (N, H, W) in one example, N is number
        # of objects.
        segms = np.zeros((num_obj, im_height, im_width), dtype='float32')
        # Information of the objects.
        obj_info = []

        # Compute images and segmentation maps (unoccluded).
        for jj, obj in enumerate(ex):
            radius = obj['radius']
            center = obj['center']
            typ = obj['type']
            segm = np.zeros((im_height, im_width), dtype='float32')
            if typ == 0:
                _draw_circle(img, center, radius, fill_color,
                             border_color, thickness)
                _draw_circle(segm, center, radius, segm_color)
            elif typ == 1:
                # Make triangles look larger.
                radius *= 1.2
                obj['radius'] = radius
                _draw_triangle(img, center, radius, fill_color,
                               border_color, thickness)
                _draw_triangle(segm, center, radius, segm_color)
            elif typ == 2:
                _draw_square(img, center, radius, fill_color,
                             border_color, thickness)
                _draw_square(segm, center, radius, segm_color)
            else:
                raise Exception('Unknown object type: {}'.format(typ))

            segms[jj] = segm
            obj_info.append(obj)

        if num_obj > 1:
            for jj in xrange(num_obj - 2, -1, -1):
                mask = np.logical_not(segms[jj + 1:, :, :].any(axis=0))
                # cv2.imshow('mask {}'.format(jj), mask.astype('float32'))
                log.info('Before sum: {}'.format(segms[jj].sum()), verbose=2)
                log.info('Top mask sum: {}'.format(mask.sum()), verbose=2)
                segms[jj] = np.logical_and(segms[jj], mask).astype('float32')
                log.info('After sum: {}'.format(segms[jj].sum()), verbose=2)

        # Aggregate results.
        # cv2.waitKey()
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
    label_objectness = np.zeros((num_ex_final, 1), dtype='float32')

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
            xx = max(center[0] + dx, size / 2)
            yy = max(center[1] + dy, size / 2)
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
            resize_imag = cv2.resize(crop_imag, output_window_size) / 255.0
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
                resize_imag = cv2.resize(crop_imag, output_window_size) / 255.0
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


def _plot_segmentation_data(img, segm_label, obj_label, title=''):
    num_row = 10
    num_col = 2
    f, axarr = plt.subplots(num_row, num_col)

    for ii in xrange(num_row):
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
        axarr[ii, 0].imshow(img[ii])
        axarr[ii, 1].imshow(segm_label[ii])
        axarr[ii, 1].text(0, 0, '{:.2f}'.format(obj_label[ii, 0]),
                          color=(0, 0, 0), size=8)
    f.suptitle(title, fontsize=16)

    return f, axarr


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
    parser.add_argument('-border_thickness', default=kBorderThickness,
                        type=int, help='Object border thickness')
    parser.add_argument('-num_ex', default=kNumExamples, type=int,
                        help='Number of examples')
    parser.add_argument('-max_num_objects', default=kMaxNumObjects, type=int,
                        help='Maximum number of circles')
    parser.add_argument('-num_object_types', default=kNumObjectTypes, type=int,
                        help='Number of object types')
    parser.add_argument('-neg_pos_ratio', default=kNegPosRatio, type=int,
                        help='Ratio between negative and positive examples')
    parser.add_argument('-min_window_size', default=kMinWindowSize, type=int,
                        help='Minimum window size to crop')
    parser.add_argument('-output_window_size', default=kOutputWindowSize,
                        type=int, help='Output segmentation window size')
    parser.add_argument('-output', default=None, help='Output file name')
    parser.add_argument('-plot', action='store_true',
                        help='Whether to plot generated data.')
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
        'border_thickness': args.border_thickness,
        'num_examples': args.num_ex,
        'max_num_objects': args.max_num_objects,
        'num_object_types': args.num_object_types,
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

    segm_data = get_segmentation_data(opt, image_data)
    log.info('Segmentation examples: {}'.format(len(segm_data['input'])))
    log.info('Segmentation input: {}'.format(segm_data['input'].shape))
    log.info('Segmentation label: {}'.format(
        segm_data['label_segmentation'].shape))

    if args.plot:
        num_row = 5
        num_col = 4

        # Plot images
        f1, axarr1 = plt.subplots(num_row, num_col)
        f1.suptitle('Full Images')
        for ii in xrange(num_row * num_col):
            row = ii / num_col
            col = ii % num_col
            img = image_data['images'][ii]
            axarr1[row, col].imshow(img)
            axarr1[row, col].set_axis_off()

        # Plot segmentations
        f2, axarr2 = plt.subplots(num_row, num_col)
        f2.suptitle('Instance Segmentation Maps')
        for ii in xrange(num_row):
            num_obj = image_data['segmentations'][ii].shape[0]
            for jj in xrange(num_col):
                if jj == 0:
                    img = image_data['images'][ii]
                    axarr2[ii, jj].imshow(img)
                elif jj <= num_obj:
                    img = image_data['segmentations'][ii][jj - 1]
                    axarr2[ii, jj].imshow(img)
                axarr2[ii, jj].set_axis_off()

        # Plot segmentation training data
        img = segm_data['input']
        segm_label = segm_data['label_segmentation']
        obj_label = segm_data['label_objectness']

        # Plot positive and negative examples
        pos_idx = obj_label[:, 0] == 1
        neg_idx = obj_label[:, 0] == 0

        f3, axarr3 = _plot_segmentation_data(img[pos_idx], segm_label[
            pos_idx], obj_label[pos_idx],
            title='Postive Training Examples')
        f3, axarr3 = _plot_segmentation_data(img[neg_idx], segm_label[
            neg_idx], obj_label[neg_idx],
            title='Negative Training Examples')

        plt.show()
