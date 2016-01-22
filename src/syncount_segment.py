import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.5.0/lib/python2.7/site-packages')
# sys.path.insert(0, '/pkgs/tensorflow-cpu-0.5.0')

from utils import logger
from utils.batch_iter import BatchIterator
from utils.grad_clip_optim import GradientClipOptimizer
from utils.time_series_logger import TimeSeriesLogger
import argparse
import datetime
import numpy as np
import os
import pickle as pkl
import syncount_gen_data as data
import tensorflow as tf
import time

log = logger.get()


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


def conv2d(x, W):
    """2-D convolution."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_4x4(x):
    """4 x 4 average pooling."""
    return tf.nn.avg_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME')


def weight_variable(shape):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def get_inference_model(opt, sess, train_model, device='/cpu:0'):
    """Get a model (for inference). Copy weights from trained model.
    Apply model densely on all sliding windows.

    Args:
        opt: dictionary, model options
        sess: tensorflow session
        train_model: restored train model for assigning weights
        device: device to store inference model
    Returns:
        model: dictionary.
    """
    # The inference network is fully convolutional so it is not dependent on
    # the input image size.
    # inp_width = opt['width']
    # inp_height = opt['height']

    with tf.device(device):
        # Model input
        # (batch, inp_height, inp_width, 3)
        inp = tf.placeholder('float', [None, None, None, 3])

        # 1st convolution layer
        # (batch, inp_height / 2, inp_width / 2, 16)
        w_conv1 = tf.constant(sess.run(train_model['w_conv1']))
        b_conv1 = tf.constant(sess.run(train_model['b_conv1']))
        h_conv1 = tf.nn.relu(conv2d(inp, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # 2nd convolution layer
        # (batch, inp_height / 4, inp_width / 4, 32)
        w_conv2 = tf.constant(sess.run(train_model['w_conv2']))
        b_conv2 = tf.constant(sess.run(train_model['b_conv2']))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # 3rd convolution layer
        # (batch, inp_height / 8, inp_width / 8, 64)
        w_conv3 = tf.constant(sess.run(train_model['w_conv3']))
        b_conv3 = tf.constant(sess.run(train_model['b_conv3']))
        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        # Helper sizes
        # out_conv_size_h = inp_height / 8
        # out_conv_size_w = inp_width / 8
        # Dense layer (apply convolution on all sliding windows)
        # (batch, out_conv_size_h, out_conv_size_w, lo_size * lo_size)

        # Output map of the train network.
        out_size = opt['output_window_size']

        # Convolution filter size of the train network.
        conv_size = out_size / 8

        lo_size = out_size / opt['output_downsample']
        w_fc1_val = np.reshape(sess.run(train_model['w_fc1']),
                               [conv_size, conv_size, 64, lo_size * lo_size])
        w_fc1 = tf.constant(w_fc1_val)
        b_fc1 = tf.constant(sess.run(train_model['b_fc1']))
        segm_reshape = tf.sigmoid(conv2d(h_pool3, w_fc1) + b_fc1)

        # (batch * out_conv_size_h * out_conv_size_w, lo_size, lo_size , 1)
        segm_lo = tf.reshape(segm_reshape, [-1, lo_size, lo_size, 1])

        # Upsample to high-resolution if necessary
        # (batch * out_conv_size_h * out_conv_size_w, out_size, out_size, 1)
        if opt['output_downsample'] > 1:
            segm_hi = tf.image.resize_nearest_neighbor(
                segm_lo, (out_size, out_size))
        else:
            segm_hi = segm_lo

        # Final output
        # (batch * out_conv_size_h * out_conv_size_w, out_size, out_size)
        segm = tf.reshape(segm_hi, [-1, out_size, out_size])

        # Objectness network
        # (batch * out_conv_size_h * out_conv_size_w, 1)
        w_fc2_val = np.reshape(sess.run(train_model['w_fc2']),
                               [conv_size, conv_size, 64, 1])
        w_fc2 = tf.constant(w_fc2_val)
        b_fc2 = tf.constant(sess.run(train_model['b_fc2']))

        # (batch * out_conv_size_h * out_conv_size_w, 1)
        obj = tf.sigmoid(conv2d(h_pool3, w_fc2) + b_fc2)

    model = {
        'inp': inp,
        'h_pool1': h_pool1,
        'h_conv2': h_conv2,
        'h_pool2': h_pool2,
        'h_pool3': h_pool3,
        'segm_lo': segm_lo,
        'segm_hi': segm_hi,
        'segm': segm,
        'obj': obj
    }

    return model


def get_train_model(opt, device='/cpu:0'):
    """Get a model (for training).

    Args:
        opt: dictionary, model options
    Returns:
        model: dictionary.
    """
    out_size = opt['output_window_size']

    with tf.device(device):
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

        # After 3 convolutional layers, the image has been downsampled by a
        # factor of 8. conv_size is the output image size of the 3rd
        # convolution layer.
        conv_size = out_size / 8
        log.info('Conv size {}'.format(conv_size))
        full_size = conv_size * conv_size * 64
        log.info('Full size {}'.format(full_size))
        h_pool3_reshape = tf.reshape(h_pool3, [-1, full_size])

        # Segmentation network
        lo_size = out_size / opt['output_downsample']
        w_fc1 = weight_variable([full_size, lo_size * lo_size])
        b_fc1 = weight_variable([lo_size * lo_size])

        # Output low-resolution image
        segm_reshape = tf.sigmoid(tf.matmul(h_pool3_reshape, w_fc1) + b_fc1)
        segm_lo = tf.reshape(segm_reshape, [-1, lo_size, lo_size, 1])

        # Upsample to high-resolution if necessary
        if opt['output_downsample'] > 1:
            segm_hi = tf.image.resize_nearest_neighbor(
                segm_lo, (out_size, out_size))
        else:
            segm_hi = segm_lo

        # Segmentation label
        segm_gt = tf.placeholder('float', [None, out_size, out_size])
        segm_gt_reshape = tf.reshape(segm_gt, [-1, out_size, out_size, 1])

        if opt['output_downsample'] > 1:
            segm_gt_lo = avg_pool_4x4(segm_gt_reshape)
        else:
            segm_gt_lo = segm_gt_reshape

        # Final output
        segm = tf.reshape(segm_hi, [-1, out_size, out_size])

        # CE average accross all image pixels (low-res version).
        segm_ce = -tf.reduce_sum(
            segm_gt_lo * tf.log(segm_lo + 1e-7) +
            (1 - segm_gt_lo) * tf.log(1 - segm_lo + 1e-7),
            reduction_indices=[1, 2])

        # Objectness network
        w_fc2 = weight_variable([full_size, 1])
        b_fc2 = weight_variable([1])
        obj = tf.sigmoid(tf.matmul(h_pool3_reshape, w_fc2) + b_fc2)
        obj_gt = tf.placeholder('float', [None, 1])

        obj_ce = -tf.reduce_sum(obj_gt * tf.log(obj + 1e-7) +
                                (1 - obj_gt) * tf.log(1 - obj + 1e-7)) / \
            tf.to_float(tf.size(obj_gt))

        # Constants
        # Mixing coefficient of objectness and segmentation objectives
        r = lo_size
        # Learning rate
        lr = 1e-4
        # Small constant for numerical stability
        eps = 1e-7

        # Total objective
        # Only count error for segmentation when there is a positive example.
        total_err = tf.reduce_sum(segm_ce * obj_gt) / \
            tf.to_float(tf.size(obj_gt))
        total_err += r * obj_ce
        # train_step = GradientClipOptimizer(
        #     tf.train.AdamOptimizer(lr, epsilon=eps)).minimize(total_err)
        train_step = tf.train.AdamOptimizer(
            lr, epsilon=eps).minimize(total_err)

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
        'w_fc1': w_fc1,
        'b_fc1': b_fc1,
        'segm_lo': segm_lo,
        'segm_hi': segm_hi,
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


def preprocess(inp, label_segmentation, label_objectness):
    """Preprocess training data."""
    return inp.astype('float32') / 255, label_segmentation.astype('float32'), \
        label_objectness.astype('float32')


def add_catalog(results_folder, model_id):
    """Add catalog entry.
    (Should refractor this logic outside training file)
    """
    with open(os.path.join(results_folder, 'catalog'), 'a') as f:
        f.write('{}\n'.format(model_id))

    pass


def save_ckpt(folder, sess, opt, global_step=None):
    """Save checkpoint.

    Args:
        folder:
        sess:
        global_step:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    ckpt_path = os.path.join(
        folder, 'model.ckpt'.format(model_id))
    log.info('Saving checkpoint to {}'.format(ckpt_path))
    saver.save(sess, ckpt_path, global_step=global_step)
    opt_path = os.path.join(folder, 'opt.pkl')
    with open(opt_path, 'wb') as f_opt:
        pkl.dump(opt, f_opt)

    pass


def parse_args():
    """Parse input arguments."""

    # Default constant parameters
    # Full image height.
    kHeight = 224
    # Full image width.
    kWidth = 224
    # Object radius lower bound.
    kRadiusLower = 5
    # Object radius upper bound.
    kRadiusUpper = 20
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
    kSizeVar = 20
    # Random window center variance.
    kCenterVar = 20
    # Resample window size (segmentation output unisize).
    kOutputWindowSize = 128
    # Ratio of negative and positive examples for segmentation data.
    kNegPosRatio = 5
    # Minimum window size of the original image (before upsampling).
    kMinWindowSize = 20
    # Downsample ratio of output.
    kOutputDownsample = 4
    # Number of steps
    kNumSteps = 2000
    # Number of steps per checkpoint
    kStepsPerCkpt = 1000

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
    parser.add_argument('-border_thickness', default=kBorderThickness,
                        type=int, help='Object border thickness')
    parser.add_argument('-num_ex', default=kNumExamples, type=int,
                        help='Number of examples')
    parser.add_argument('-max_num_objects', default=kMaxNumObjects, type=int,
                        help='Maximum number of objects')
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
    parser.add_argument('-output_downsample', default=kOutputDownsample,
                        type=int, help='Output downsample ratio')
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-results', default='../results',
                        help='Model results folder')
    parser.add_argument('-logs', default='../results',
                        help='Training curve logs folder')
    parser.add_argument('-localhost', default='localhost',
                        help='Local domain name')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    log.log_args()

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    # Model options
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
        'output_window_size': args.output_window_size,
        'output_downsample': args.output_downsample
    }

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    # Logistics
    results_folder = args.results
    logs_folder = args.logs
    task_name = 'syncount_segment'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)
    exp_folder = os.path.join(results_folder, model_id)
    exp_logs_folder = os.path.join(logs_folder, model_id)

    # Create model
    m = get_train_model(opt, device=device)
    log.info('All variables: {}'.format([x.name for x in tf.all_variables()]))
    log.info('Trainable variables: {}'.format(
        [x.name for x in tf.trainable_variables()]))

    # Initialize session
    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())

    # Fetch dataset
    # You probably want to store a disk-version of the dataset if it is very large.
    # You probably want to flush to disk while generating data.
    dataset = get_dataset(opt, opt['num_examples'], opt['num_examples'] / 10)

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

    # Create time series logger
    train_ce_logger = TimeSeriesLogger(
        os.path.join(exp_logs_folder, 'train_ce.csv'), 'train_ce',
        buffer_size=25)
    valid_ce_logger = TimeSeriesLogger(
        os.path.join(exp_logs_folder, 'valid_ce.csv'), 'valid_ce',
        buffer_size=2)
    log.info(
        'Curves can be viewed at: http://{}/visualizer?id={}'.format(
            args.localhost, model_id))

    step = 0
    while step < loop_config['num_steps']:
        # Validation
        valid_ce = 0
        for st, nd in BatchIterator(num_ex_val, batch_size=64, progress_bar=False):
            inp_batch = inp_all_val[st: nd]
            lab_seg_batch = lab_seg_all_val[st: nd]
            lab_obj_batch = lab_obj_all_val[st: nd]
            inp_batch, lab_seg_batch, lab_obj_batch = preprocess(
                inp_batch, lab_seg_batch, lab_obj_batch)
            vce = sess.run(m['total_err'],
                           feed_dict={m['inp']: inp_batch,
                                      m['segm_gt']: lab_seg_batch,
                                      m['obj_gt']: lab_obj_batch})
            valid_ce += vce * inp_batch.shape[0] / float(num_ex_val)
        
        log.info('{:d} valid ce: {:.4f}'.format(step, valid_ce))
        valid_ce_logger.add(step, valid_ce)

        # Train
        for st, nd in BatchIterator(num_ex, batch_size=64, progress_bar=False):
            inp_batch = inp_all[st: nd]
            lab_seg_batch = lab_seg_all[st: nd]
            lab_obj_batch = lab_obj_all[st: nd]
            inp_batch, lab_seg_batch, lab_obj_batch = preprocess(
                inp_batch, lab_seg_batch, lab_obj_batch)
            tim = time.time()
            r = sess.run([m['total_err'], m['train_step']],
                         feed_dict={m['inp']: inp_batch,
                                    m['segm_gt']: lab_seg_batch,
                                    m['obj_gt']: lab_obj_batch})
            train_ce = r[0]
            if step % 10 == 0:
                log.info('{:d} ce {:.4f} t {:.2f}ms'.format(
                    step, train_ce, time.time() - tim))
                train_ce_logger.add(step, train_ce)
            step += 1

            # Save model
            if step % loop_config['steps_per_ckpt'] == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)
                add_catalog(results_folder, model_id)
