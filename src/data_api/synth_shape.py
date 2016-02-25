"""
Generate synthetic images with circles, triangles, and squares.

Output:
* Images: [H, W]
* Groundtruth segmentations for each object: [M, H, W] (M objects)

Usage: python syncount_gen_data.py --help
"""

from utils import logger
from utils import progress_bar
from utils.sharded_hdf5 import ShardedFile, ShardedFileWriter
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Logger
log = logger.get()


def get_raw_data(opt, seed=2):
    """Generate raw data (dictionary).

    Args:
        opt: dictionary, options.
        seed: int, rng seed.
    Returns:
        results:
    """
    num_examples = opt['num_examples']
    max_num_objects = opt['max_num_objects']
    radius_lower = opt['radius_lower']
    radius_upper = opt['radius_upper']
    width = opt['width']
    height = opt['height']
    num_object_types = opt['num_object_types']
    results = []
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
            obj_type = int(
                np.floor(random.uniform(0, num_object_types - 1e-5)))
            ex.append({
                'center': center,
                'radius': radius,
                'type': obj_type
            })

        results.append(ex)

    return results


def _draw_circle(img, center, radius, fill, border=None, thickness=None):
    """Draw a circle given center and radius.

    Args:
        img: numpy.ndarray, [H, W, 3], dtype=uint8, [0, 255]
        center: tuple, (x, y)
        radius: float
        fill: tuple, (B, G, R)
        border: tuple, (B, G, R)
        thickness: float
    """
    cv2.circle(img, center=center, radius=radius, color=fill, thickness=-1)

    if border:
        cv2.circle(img, center=center, radius=radius,
                   color=border, thickness=thickness)

    pass


def _draw_square(img, center, radius, fill, border=None, thickness=None):
    """Draw a square given center and radius.

    Args:
        img: numpy.ndarray, [H, W, 3], dtype=uint8, [0, 255]
        center: tuple, (x, y)
        radius: float
        fill: tuple, (B, G, R)
        border: tuple, (B, G, R)
        thickness: float
    """
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
    """Draw a equilateral triangle given center and radius.

    Args:
        img: numpy.ndarray, [H, W, 3], dtype=uint8, [0, 255]
        center: tuple, (x, y)
        radius: float
        fill: tuple, (B, G, R)
        border: tuple, (B, G, R)
        thickness: float
    """
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


def _raw_to_image(opt, raw_data_entry):
    """Convert a raw data entry (object info) to an image data entry
    (rasterized image).

    Args:
        raw_data_entry: list
    Returns:
        image_data: dictionary
            image: numpy.ndarray, [H, W, 3], dtype=uint8, [0, 255]
            segmentations: numpy.ndarray, [M, H, W], dtype=uint8, [0, 1]
            object_info: dictionary.
    """
    im_height = opt['height']
    im_width = opt['width']
    thickness = opt['border_thickness']
    # Draw a green objects with red border.
    # Note: OpenCV uses (B, G, R) order.
    fill_color = (0, 255, 0)
    border_color = (0, 0, 255)
    segm_color = (1, 0, 0)

    img = np.zeros([im_height, im_width, 3], dtype='uint8')
    num_obj = len(raw_data_entry)
    # Segmentations of all objects (N, H, W) in one example, N is number
    # of objects.
    segms = np.zeros([num_obj, im_height, im_width], dtype='uint8')
    # Information of the objects.
    obj_info = []

    # Compute images and segmentation maps (unoccluded).
    for jj, obj in enumerate(raw_data_entry):
        radius = obj['radius']
        center = obj['center']
        typ = obj['type']
        segm = np.zeros([im_height, im_width], dtype='uint8')
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

    # Apply occlusion.
    if num_obj > 1:
        for jj in xrange(num_obj - 2, -1, -1):
            mask = np.logical_not(segms[jj + 1:, :, :].any(axis=0))
            # cv2.imshow('mask {}'.format(jj), mask.astype('float32'))
            log.info('Before sum: {}'.format(segms[jj].sum()), verbose=2)
            log.info('Top mask sum: {}'.format(mask.sum()), verbose=2)
            segms[jj] = np.logical_and(segms[jj], mask).astype('uint8')
            log.info('After sum: {}'.format(segms[jj].sum()), verbose=2)

    # Aggregate results.
    return {
        'image': img,
        'segmentations': segms,
        'object_info': obj_info
    }


def get_image_data(opt, raw_data):
    """Compile image and segmentation maps with object information.

    Args:
        opt
        raw_data
    Returns:
        image_data: dictionary. Contains following fields:
            images: list of numpy.ndarray, [H, W, 3], H is
            image height. W is image width. Each image has RGB 3 channels.
            segmentations: list of numpy.ndarray, each item has shape
            (M, H, W). M is number of objects. H is image height. W is image
            width. Binary mask.
            object_info: dictionary.
    """
    im_height = opt['height']
    im_width = opt['width']
    image_data = []

    log.info('Compiling images, {} examples'.format(len(raw_data)))
    for raw_data_entry in progress_bar.get_iter(raw_data):
        image_data.append(_raw_to_image(opt, raw_data_entry))

    return image_data


def _image_to_segmentation(opt, image_data_entry, random=None):
    """Convert image_data_entry to segmentation training data

    Args:
        opt: list of dictionary
        image_data_entry: dictionary
        random: numpy.random.RandomState object
    Returns:
        results: list of dictionary. Each has the following fields:
            input: numpy.ndarray, [Hp, Wp, 3], dtype=uint8, [0, 255]
            label_segmentation: numpy.ndarray, [Hp, Wp], dtype=uint8, [0, 1]
            label_objectness: float, [0, 1]
    """

    if random is None:
        random = np.random.RandomState(2)

    min_window_size = opt['min_window_size']
    neg_pos_ratio = opt['neg_pos_ratio']
    radius_upper = opt['radius_upper']
    outsize = opt['output_window_size']
    output_window_size = (outsize, outsize)
    full_height = image_data_entry['image'].shape[0]
    full_width = image_data_entry['image'].shape[1]
    obj_info = image_data_entry['object_info']
    num_ex_final = len(obj_info) * (1 + neg_pos_ratio)

    # Output
    results = []

    for jj, obj in enumerate(obj_info):
        image = image_data_entry['image']
        segmentations = image_data_entry['segmentations']
        center = obj['center']
        radius = obj['radius']

        # Gittering on center.
        dx = int(np.ceil(random.normal(0, opt['center_var'])))
        dy = int(np.ceil(random.normal(0, opt['center_var'])))

        # Random scaling, window size (square shape).
        size = max(
            4 * radius + int(np.ceil(random.normal(0, opt['size_var']))),
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
        crop_imag = image[yy - size / 2: yy + size / 2,
                          xx - size / 2: xx + size / 2]
        crop_segm = segmentations[
            jj, yy - size / 2: yy + size / 2, xx - size / 2: xx + size / 2]
        log.info('Cropped image size: {}'.format(crop_imag.shape),
                 verbose=2)
        log.info('Resized image size: {}'.format(output_window_size),
                 verbose=2)

        # Resample the image and segmentation to be uni-size.
        if crop_imag.size > 0:
            resize_imag = cv2.resize(crop_imag, output_window_size)
            resize_segm = cv2.resize(crop_segm, output_window_size)
            results.append({
                'input': resize_imag,
                'label_segmentation': resize_segm,
                'label_objectness': 1
            })

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

            crop_imag = image[y: y + size, x: x + size]
            if crop_imag.size > 0:
                resize_imag = cv2.resize(crop_imag, output_window_size)
                resize_segm = np.zeros(output_window_size, dtype='uint8')
                results.append({
                    'input': resize_imag,
                    'label_segmentation': resize_segm,
                    'label_objectness': 0
                })

    return results


def write_segmentation_data(opt, image_data, fname, num_per_shard=16000,
                            seed=2):
    """Write segmentation network training data to sharded file.

    Args:
        opt: dictionary
        image_data: list of dictionary
        fname: string, output file basename
        num_per_shard: int, number of training examples per file shard
        seed: rng seed
    """

    random = np.random.RandomState(seed)
    neg_pos_ratio = opt['neg_pos_ratio']

    # Count total number of examples.
    num_ex = len(image_data)
    num_ex_final = 0
    for ii in xrange(num_ex):
        num_ex_final += len(image_data[ii]['object_info'])
    num_ex_final *= (1 + neg_pos_ratio)
    log.info('Preparing segmentation data, {} examples'.format(num_ex_final))

    num_shards = int(np.ceil(num_ex_final / float(num_per_shard)))
    log.info('Writing to {} in {} shards'.format(fname, num_shards))

    fout = ShardedFile(fname, num_shards=num_shards)
    with ShardedFileWriter(fout, num_ex_final) as writer:
        for ii in progress_bar.get(num_ex):
            segm_data = _image_to_segmentation(
                opt, image_data[ii], random=random)
            for segm_j in segm_data:
                writer.write(segm_j)

    pass


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
        results: list of dictionary. Each has the following fields:
            input: numpy.ndarray, [Hp, Wp, 3], dtype=uint8, [0, 255]
            label_segmentation: numpy.ndarray, [Hp, Wp], dtype=uint8, [0, 1]
            label_objectness: float, [0, 1]
    """
    random = np.random.RandomState(seed)
    neg_pos_ratio = opt['neg_pos_ratio']
    outsize = opt['output_window_size']

    # Count total number of examples.
    num_ex = len(image_data)
    num_ex_final = 0
    for ii in xrange(num_ex):
        num_ex_final += len(image_data[ii]['object_info'])
    num_ex_final *= (1 + neg_pos_ratio)
    log.info('Preparing segmentation data, {} examples'.format(num_ex_final))

    # Initialize arrays.
    input_data = np.zeros([num_ex_final, outsize, outsize, 3], dtype='uint8')
    label_segmentation = np.zeros(
        [num_ex_final, outsize, outsize], dtype='uint8')
    label_objectness = np.zeros([num_ex_final, 1], dtype='uint8')

    idx = 0
    for ii in progress_bar.get(num_ex):
        segm_data = _image_to_segmentation(opt, image_data[ii], random=random)
        for segm_j in segm_data:
            input_data[idx] = segm_j['input']
            label_segmentation[idx] = segm_j['label_segmentation']
            label_objectness[idx] = segm_j['label_objectness']
            idx += 1

    return {
        'input': input_data,
        'label_segmentation': label_segmentation,
        'label_objectness': label_objectness
    }


def _add_noise_to_segmentation(image_data_entry, seed=2):
    """Add noise to segmentation labels for training counting network.
    Possibly duplicated noisy segmentation labels for a single instance.
    Number of copies is uniform from 1 to 10.
    Shift the object by random move.
    White noise (Ideally we want deformation + white noise)

    Args:
        image_data_entry
    """

    random = np.random.RandomState(seed)
    segmentations = []

    # Parameters of a noisy linear transformation.
    trans_std = 0.02
    rotation_std = 0.02
    height = image_data_entry['image'].shape[0]
    width = image_data_entry['image'].shape[1]

    for segm in image_data_entry['segmentations']:
        num_copies = int(np.floor(random.uniform(1, 11 - 1e-5)))
        num_pts = (segm > 0).astype('int32').sum()
        pts_arr = np.array(segm.nonzero()).transpose()
        pts_arr = np.concatenate(
            [pts_arr, np.ones([pts_arr.shape[0], 1])], axis=1)

        # Apply random linear transformation that is similar to an identity
        # map.
        for ii in xrange(num_copies):
            noise_segm = np.zeros(segm.shape, dtype='uint8')
            lintrans = np.eye(3)
            lintrans[2, 0: 2] += random.normal(0, trans_std, [2])
            lintrans[0: 2, 0: 2] += random.normal(0, rotation_std, [2, 2])
            lintrans[0, 2] = 0
            lintrans[1, 2] = 0
            lintrans[2, 2] = 1
            pts_arr_img = pts_arr.dot(lintrans)
            pts_arr_img = pts_arr_img.astype('int32')
            valid_idx_x = np.logical_and(
                pts_arr_img[:, 0] >= 0, pts_arr_img[:, 0] < segm.shape[0])
            valid_idx_y = np.logical_and(
                pts_arr_img[:, 1] >= 0, pts_arr_img[:, 1] < segm.shape[1])
            pts_idx = pts_arr_img[np.logical_and(
                valid_idx_x, valid_idx_y), 0: 2]

            if pts_idx.shape[0] > 0:
                pts_idx = [pts_idx[:, 0], pts_idx[:, 1]]
                noise_segm[pts_idx] = 1
                segmentations.append(noise_segm)

    image_data_entry['segmentations'] = np.array(segmentations)

    pass


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


def _image_to_instance_segmentation(opt, image_data_entry):
    """
    Convert from image data to instance segmentation training data.
    """
    height = opt['height']
    width = opt['width']
    timespan = opt['max_num_objects'] + 1
    image = image_data_entry['image']
    ins_segm = np.zeros([timespan, height, width], dtype='uint8')
    num_segmentations = len(image_data_entry['segmentations'])
    for tt, segm in enumerate(image_data_entry['segmentations']):
        ins_segm[tt] = image_data_entry['segmentations'][tt]

    return {
        'image': image,
        'segmentations': ins_segm,
        'num_segmentations': num_segmentations
    }


def get_instance_segmentation_data(opt, image_data):
    """
    Gets instance segmentation data.

    Args:
        opt
        image_data
    """
    num_ex = len(image_data)
    height = opt['height']
    width = opt['width']
    timespan = opt['max_num_objects'] + 1
    inp = np.zeros([num_ex, height, width, 3], dtype='uint8')
    label_segmentation = np.zeros(
        [num_ex, timespan, height, width], dtype='uint8')
    label_score = np.zeros([num_ex, timespan], dtype='uint8')
    for ii, image_data_entry in enumerate(image_data):
        ins_segm = _image_to_instance_segmentation(opt, image_data_entry)
        inp[ii] = ins_segm['image']
        label_segmentation[ii] = ins_segm['segmentations']
        label_score[ii, :ins_segm['num_segmentations']] = 1

    return {
        'input': inp,
        'label_segmentation': label_segmentation,
        'label_score': label_score
    }


def get_dataset(opt, seed=2):
    raw_data = get_raw_data(opt, seed=seed)
    image_data = get_image_data(opt, raw_data)
    segm_data = get_instance_segmentation_data(opt, image_data)

    return segm_data


def parse_args():
    """Parse input arguments."""
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
    parser.add_argument('-center_var', default=kCenterVar, type=float,
                        help='Image patch center variance')
    parser.add_argument('-size_var', default=kSizeVar, type=float,
                        help='Image patch size variance')
    parser.add_argument('-neg_pos_ratio', default=kNegPosRatio, type=int,
                        help='Ratio between negative and positive examples')
    parser.add_argument('-min_window_size', default=kMinWindowSize, type=int,
                        help='Minimum window size to crop')
    parser.add_argument('-output_window_size', default=kOutputWindowSize,
                        type=int, help='Output segmentation window size')
    parser.add_argument('-output', default=None, help='Output file name')
    parser.add_argument('-noise', action='store_true',
                        help='Whether to add noise to image data.')
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
        'center_var': args.center_var,
        'size_var': args.size_var,
        'neg_pos_ratio': args.neg_pos_ratio,
        'min_window_size': args.min_window_size,
        'output_window_size': args.output_window_size
    }
    raw_data = get_raw_data(opt)
    image_data = get_image_data(opt, raw_data)
    log.info('Images: {}'.format(len(image_data)))
    log.info('Segmentation 1: {}'.format(image_data[0]['segmentations'].shape))

    # Add noise to segmentations.
    if args.noise:
        for image_data_entry in progress_bar.get_iter(image_data):
            _add_noise_to_segmentation(image_data_entry)

    segm_data = get_segmentation_data(opt, image_data)
    log.info('Segmentation examples: {}'.format(len(segm_data['input'])))
    log.info('Segmentation input: {}'.format(segm_data['input'].shape))
    log.info('Segmentation label: {}'.format(
        segm_data['label_segmentation'].shape))

    ins_segm_data = get_instance_segmentation_data(opt, image_data)
    log.info('Instance segmentation input: {}'.format(
        ins_segm_data['input'].shape))
    log.info('Instance segmentation label: {}'.format(
        ins_segm_data['label_segmentation'].shape))
    log.info(ins_segm_data['input'][0])
    log.info(ins_segm_data['label_segmentation'][0][0])
    log.info(ins_segm_data['label_score'][0])

    # Write training data to file.
    if args.output:
        write_segmentation_data(opt, image_data, args.output)

    if args.plot:
        num_row = 5
        num_col = 4

        # Plot images
        f1, axarr1 = plt.subplots(num_row, num_col)
        f1.suptitle('Full Images')
        for ii in xrange(num_row * num_col):
            row = ii / num_col
            col = ii % num_col
            img = image_data[ii]['image']
            axarr1[row, col].imshow(img)
            axarr1[row, col].set_axis_off()

        # Plot segmentations
        f2, axarr2 = plt.subplots(num_row, num_col)
        f2.suptitle('Instance Segmentation Maps')
        for ii in xrange(num_row):
            num_obj = image_data[ii]['segmentations'].shape[0]
            for jj in xrange(num_col):
                if jj == 0:
                    img = image_data[ii]['image']
                    axarr2[ii, jj].imshow(img)
                elif jj <= num_obj:
                    img = image_data[ii]['segmentations'][jj - 1]
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
