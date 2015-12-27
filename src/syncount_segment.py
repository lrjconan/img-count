import sys
sys.path.insert(0, '/pkgs/tensorflow-cpu-0.5.0')
from utils import logger
import argparse
import syncount_gen_data as data
import tensorflow as tf

log = logger.get()

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


def get_dataset(opt, num_train, num_valid):
    """Get train-valid split dataset for instance segmentation.

    Args:
        opt
        num_train
        num_valid
    Returns:
        dataset
            train
            valid
    """
    dataset = {}
    opt['num_examples'] = num_train
    raw_data = data.get_raw_data(opt, seed=2)
    image_data = data.get_image_data(opt, raw_data)
    segm_data = data.get_segmentation_data(opt, image_data, seed=2)
    dataset['train'] = segm_data

    opt['num_examples'] = num_valid
    raw_data = data.get_raw_data(opt, seed=3)
    image_data = data.get_image_data(opt, raw_data)
    segm_data = data.get_segmentation_data(opt, image_data, seed=3)
    dataset['valid'] = segm_data

    return dataset


def create_model(opt):
    pass


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
    # Options
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

    # Create model
    imwidth = opt['width']
    imheight = opt['height']
    x = tf.placeholder('float', [None, imwidth, imheight])
    out_size = opt['output_window_size']

    x_reshape = tf.reshape(x, [-1, imwidth * imheight])
    weights = tf.Variable(tf.zeros([imwidth * imheight, out_size * out_size]))
    bias = tf.Variable(tf.zeros([out_size * out_size]))
    y_reshape = tf.matmul(x_reshape, weights) + bias
    y = tf.reshape(y_reshape, [-1, out_size, out_size])
    y_ = tf.placeholder('float', [None, out_size, out_size])
    mse = tf.reduce_mean(0.5 * (y - y_) * (y - y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)

    # Start training
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    dataset = get_dataset(opt, 100, 100)

    for i in xrange(1000):
        xval = dataset['train']['input']
        y_val = dataset['train']['label_segmentation']
        sess.run(train_step, feed_dict={
            x: xval,
            y_: y_val
        })
        train_mse = sess.run(mse, feed_dict={
            x: xval,
            y_: y_val
        })
        log.info('step: {:d}, mse: {:4f}'.format(i, train_mse))

#     dataset.
#     kernel = tf.Variable


# with tf.variable_scope('conv1') as scope:
#     kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
#                                          stddev=1e-4, wd=0.0)
#     conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#     biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
#     bias = tf.nn.bias_add(conv, biases)
#     conv1 = tf.nn.relu(bias, name=scope.name)
#     _activation_summary(conv1)
