import sys
sys.path.insert(0, '/pkgs/tensorflow-cpu-0.5.0')

from utils import logger
from utils.batch_iter import BatchIterator

import argparse
import datetime
import numpy as np
import os
import syncount_gen_data as data
import tensorflow as tf
import pickle as pkl

log = logger.get()

# Default constant parameters
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

# Number of epochs
kNumEpochs = 5

# Number of epochs per checkpoint
kEpochsPerCkpt = 5


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


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Train models on synthetic counting images')
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
    parser.add_argument('-num_epochs', default=kNumEpochs,
                        type=int, help='Number of epochs to train')
    parser.add_argument('-epochs_per_ckpt', default=kEpochsPerCkpt,
                        type=int, help='Number of epochs per checkpoint')
    parser.add_argument('-results', default='../results',
                        help='Model results folder')
    args = parser.parse_args()

    return args


def conv2d(x, W):
    """2-D convolution."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def create_model(opt):
    """Create model.

    Args:
        opt: dictionary, model options
    Returns:
        model: dictionary.
    """
    out_size = opt['output_window_size']
    # Model input
    inp = tf.placeholder('float', [None, out_size, out_size, 3])

    # 1st convolution layer
    w_conv1 = weight_variable([5, 5, 3, 16])
    b_conv1 = weight_variable([16])
    h_conv1 = tf.nn.relu(conv2d(inp, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd convolution layer
    w_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = weight_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 3rd convolution layer
    w_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = weight_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    conv_size = 16 * 16 * 64
    h_pool3_reshape = tf.reshape(h_pool3, [-1, conv_size])

    # Output low-resolution image
    lo = False
    if lo:
        lo_size = out_size / 4
        log.info('Low resolution: {}'.format(lo_size))
        log.info('High resolution: {}'.format(out_size))

        # Segmentation network
        w_fc1 = weight_variable([conv_size, lo_size * lo_size])
        b_fc1 = weight_variable([lo_size * lo_size])
        segm_lo_reshape = tf.sigmoid(tf.matmul(h_pool3_reshape, w_fc1) + b_fc1)
        segm_lo = tf.reshape(segm_lo_reshape, [-1, lo_size, lo_size, 1])

        # Upsample
        segm_hi = tf.image.resize_bilinear(segm_lo, (out_size, out_size))
        segm_hi = tf.reshape(segm_hi, [-1, out_size, out_size])
        segm_gt = tf.placeholder('float', [None, out_size, out_size])

        # CE average accross all image pixels.
        segm_ce = -tf.reduce_sum(
            segm_gt * tf.log(segm_hi) + (1 - segm_gt) * tf.log(1 - segm_hi), 
            reduction_indices=[1, 2])
    else:
        # Segmentation network
        w_fc1 = weight_variable([conv_size, out_size * out_size])
        b_fc1 = weight_variable([out_size * out_size])
        segm_reshape = tf.sigmoid(tf.matmul(h_pool3_reshape, w_fc1) + b_fc1)
        segm = tf.reshape(segm_reshape, [-1, out_size, out_size])
        segm_gt = tf.placeholder('float', [None, out_size, out_size])

        # CE average accross all image pixels.
        segm_ce = -tf.reduce_sum(
            segm_gt * tf.log(segm) + (1 - segm_gt) * tf.log(1 - segm), 
            reduction_indices=[1, 2])

    # Objectness network
    w_fc2 = weight_variable([conv_size, 1])
    b_fc2 = weight_variable([1])
    obj = tf.sigmoid(tf.matmul(h_pool3_reshape, w_fc2) + b_fc2)
    obj_gt = tf.placeholder('float', [None, 1])
    obj_ce = -tf.reduce_mean(
        obj_gt * tf.log(obj) + (1 - obj_gt) * tf.log(1 - obj))

    # Constants
    # lbd = 1 / float(out_size * out_size)
    lbd = 1

    # Total objective
    # Only count error for segmentation when there is a positive example.
    total_err = tf.reduce_mean(segm_ce * obj_gt)
    total_err += lbd * obj_ce
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_err)

    if lo:
        model = {
            'inp': inp,
            'w_conv1': w_conv1,
            'b_conv1': b_conv1,
            'h_pool1': h_pool1,
            'w_conv2': w_conv2,
            'b_conv2': b_conv2,
            'h_conv2': h_conv2,
            'h_pool2': h_pool2,
            'b_conv3': b_conv3,
            'w_conv3': w_conv3,
            'h_pool3': h_pool3,
            'h_pool3_reshape': h_pool3_reshape,
            'w_fc1': w_fc1,
            'b_fc1': b_fc1,
            'segm_lo_reshape': segm_lo_reshape,
            'segm_lo': segm_lo,
            'segm_hi': segm_hi,
            'segm_gt': segm_gt,
            'segm_ce': segm_ce,
            'w_fc2': w_fc2,
            'b_fc2': b_fc2,
            'obj': obj,
            'obj_gt': obj_gt,
            'obj_ce': obj_ce,
            'total_err': total_err,
            'train_step': train_step
        }
    else:
        model = {
            'inp': inp,
            'w_conv1': w_conv1,
            'b_conv1': b_conv1,
            'h_pool1': h_pool1,
            'w_conv2': w_conv2,
            'b_conv2': b_conv2,
            'h_conv2': h_conv2,
            'h_pool2': h_pool2,
            'b_conv3': b_conv3,
            'w_conv3': w_conv3,
            'h_pool3': h_pool3,
            'h_pool3_reshape': h_pool3_reshape,
            'w_fc1': w_fc1,
            'b_fc1': b_fc1,
            'segm_reshape': segm_reshape,
            'segm': segm,
            'segm_gt': segm_gt,
            'segm_ce': segm_ce,
            'w_fc2': w_fc2,
            'b_fc2': b_fc2,
            'obj': obj,
            'obj_gt': obj_gt,
            'obj_ce': obj_ce,
            'total_err': total_err,
            'train_step': train_step
        }

    return model


def save_ckpt(save_folder, sess, opt, global_step=None):
    """Save checkpoint.

    Args:
        save_folder:
        sess:
        global_step:
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ckpt_path = os.path.join(
        save_folder, 'model.ckpt'.format(model_id))
    log.info('Saving checkpoint to {}'.format(ckpt_path))
    saver.save(sess, ckpt_path, global_step=global_step)
    opt_path = os.path.join(save_folder, 'opt.pkl')
    with open(opt_path, 'wb') as f_opt:
        pkl.dump(opt, f_opt)

    pass


if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    log.log_args()

    # Model options
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
    # with open('opt.pkl', 'wb') as f_opt:
    #     pkl.dump(opt, f_opt)
    # log.fatal('end')

    # Train loop options
    loop_config = {
        'num_epochs': args.num_epochs,
        'epochs_per_ckpt': args.epochs_per_ckpt
    }
  
    # Logistics
    save_folder = args.results
    task_name = 'syncount_segment'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)
    save_folder = os.path.join(save_folder, model_id)

    # Create model
    m = create_model(opt)
    log.info('All variables: {}'.format([x.name for x in tf.all_variables()]))
    log.info('Trainable variables: {}'.format(
        [x.name for x in tf.trainable_variables()]))
    # log.fatal('End')

    # Initialize session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Fetch dataset
    dataset = get_dataset(opt, 50, 20)

    inp_all = dataset['train']['input']
    lab_seg_all = dataset['train']['label_segmentation']
    lab_obj_all = dataset['train']['label_objectness']
    num_ex = inp_all.shape[0]
    log.info('{} training examples'.format(num_ex))

    inp_all_val = dataset['valid']['input']
    lab_seg_all_val = dataset['valid']['label_segmentation']
    lab_obj_all_val = dataset['valid']['label_objectness']
    num_ex_val = inp_all_val.shape[0]
    log.info('{} validation examples'.format(num_ex_val))

    # Create saver
    saver = tf.train.Saver(tf.all_variables())
    # saver = tf.train.Saver(tf.trainable_variables())

    for epoch in xrange(loop_config['num_epochs']):
        train_ce_epoch = 0
        valid_ce_epoch = 0
        log.info('Epoch {}'.format(epoch))

        # Train
        for step, idx in enumerate(BatchIterator(
                num_ex, batch_size=64,
                progress_bar=False)):
            inp_batch = inp_all[idx]
            lab_seg_batch = lab_seg_all[idx]
            lab_obj_batch = lab_obj_all[idx]
            sess.run(m['train_step'],
                     feed_dict={m['inp']: inp_batch,
                                m['segm_gt']: lab_seg_batch,
                                m['obj_gt']: lab_obj_batch})
            train_ce = sess.run(m['total_err'],
                                feed_dict={m['inp']: inp_batch,
                                           m['segm_gt']: lab_seg_batch,
                                           m['obj_gt']: lab_obj_batch})
            log.info('Step: {:d}, Train CE: {:.4f}'.format(step, train_ce))
            train_ce_epoch += train_ce * inp_batch.shape[0] / float(num_ex)

        # Validation
        for step, idx in enumerate(BatchIterator(
                num_ex_val, batch_size=64,
                progress_bar=False)):
            inp_batch = inp_all_val[idx]
            lab_seg_batch = lab_seg_all_val[idx]
            lab_obj_batch = lab_obj_all_val[idx]
            valid_ce = sess.run(m['total_err'],
                                feed_dict={m['inp']: inp_batch,
                                           m['segm_gt']: lab_seg_batch,
                                           m['obj_gt']: lab_obj_batch})
            log.info('Step: {:d}, Valid CE: {:.4f}'.format(step, valid_ce))
            valid_ce_epoch += valid_ce * inp_batch.shape[0] / float(num_ex_val)

        # Epoch statistics
        log.info('Epoch: {:d}, Train CE: {:.4f}, Valid CE {:.4f}'.format(
            epoch, train_ce_epoch, valid_ce_epoch))

        # Save model
        if (epoch + 1) % loop_config['epochs_per_ckpt'] == 0:
            save_ckpt(save_folder, sess, opt, global_step=epoch)
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            # ckpt_path = os.path.join(
            #     save_folder, 'model.ckpt'.format(model_id))
            # log.info('Saving checkpoint to {}'.format(ckpt_path))
            # saver.save(sess, ckpt_path, global_step=epoch)

    # Save options
