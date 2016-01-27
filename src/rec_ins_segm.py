"""
This code implements Recurrent Instance Segmentation [1].

Author: Mengye Ren (m.ren@cs.toronto.edu)

Usage: python rec_ins_segm.py --help

Reference:
[1] B. Romera-Paredes, P. Torr. Recurrent Instance Segmentation. arXiv preprint
arXiv:1511.08250, 2015.
"""
import cslab_environ

from data_api import mnist
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
from utils.time_series_logger import TimeSeriesLogger
import argparse
import datetime
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time

log = logger.get()


def conv2d(x, w):
    """2-D convolution."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_8x8(x):
    """8 x 8 avg pooling."""
    return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1],
                          strides=[1, 8, 8, 1], padding='SAME')


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _add_conv_lstm(model, timespan, inp_height, inp_width, inp_depth, filter_size, hid_depth, c_init, h_init, wd=None, name=''):
    """Adds a Conv-LSTM component."""
    g_i = [None] * timespan
    g_f = [None] * timespan
    g_o = [None] * timespan
    u = [None] * timespan
    c = [None] * (timespan + 1)
    h = [None] * (timespan + 1)
    c[-1] = c_init
    h[-1] = h_init

    w_xi = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           name='w_xi_{}'.format(name))
    w_hi = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           name='w_hi_{}'.format(name))
    b_i = weight_variable([hid_depth],
                          name='b_i_{}'.format(name))
    w_xf = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           name='w_xf_{}'.format(name))
    w_hf = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           name='w_hf_{}'.format(name))
    b_f = weight_variable([hid_depth],
                          name='b_f_{}'.format(name))
    w_xu = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           name='w_xu_{}'.format(name))
    w_hu = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           name='w_hu_{}'.format(name))
    b_u = weight_variable([hid_depth],
                          name='b_u_{}'.format(name))
    w_xo = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           name='w_xo_{}'.format(name))
    w_ho = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           name='w_ho_{}'.format(name))
    b_o = weight_variable([hid_depth], name='b_o_{}'.format(name))

    def unroll(inp, time):
        g_i[t] = tf.sigmoid(conv2d(inp, w_xi) + conv2d(h[t - 1], w_hi) + b_i)
        g_f[t] = tf.sigmoid(conv2d(inp, w_xf) + conv2d(h[t - 1], w_hf) + b_f)
        g_o[t] = tf.sigmoid(conv2d(inp, w_xo) + conv2d(h[t - 1], w_ho) + b_o)
        u[t] = tf.tanh(conv2d(inp, w_xu) + conv2d(h[t - 1], w_hu) + b_u)
        c[t] = g_f[t] * c[t - 1] + g_i[t] * u[t]
        h[t] = g_o[t] * tf.tanh(c[t])

        pass

    model['g_i_{}'.format(name)] = g_i
    model['g_f_{}'.format(name)] = g_f
    model['g_o_{}'.format(name)] = g_o
    model['u_{}'.format(name)] = u
    model['c_{}'.format(name)] = c
    model['h_{}'.format(name)] = h

    return unroll


def bce(y_out, y_gt):
    """
    Binary cross entropy.

    Args:
        y_out:
        y_gt:
    """
    eps = 1e-7
    return -y_gt * tf.log(y_out + eps) - (1 - y_gt) * tf.log(1 - y_out + eps)


def ins_segm_loss(y_out, y_gt, s_out, s_gt, r):
    """
    Instance segmentation loss.

    Args:
        y_out: [B, N, H, W], output segmentations.
        y_gt: [B, M, H, W], groundtruth segmentations.
        s: [B, N], confidence score.
        r: float, mixing coefficient for combining segmentation loss and 
        confidence score loss.
    """
    # IOU score, [B, N, M]
    iou = tf.user_ops.batch_iou(y_out, y_gt)
    # Matching score, [B, N, M]
    delta = tf.user_ops.hungarian(iou)
    # 
    y_gt_shape = tf.shape(y_gt)
    num_segm = tf.reshapey_gt_shape[1: 2]
    num_segm_mul = tf.concat(0, [tf.constant([1]), tf.constant([1]), num_segm])
    
    # [B, N] => [B, N, M]
    s_rep = tf.tile(tf.expand_dims(tf.cum_min(s_out), dim=3), num_segm_mul)
    return -tf.reduce_sum(iou * delta * s_gt) + r * bce(delta, s_rep)


def get_model(opt, device='/cpu:0', train=True):
    """Get model."""
    m = {}
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    conv_lstm_filter_size = opt['conv_lstm_filter_size']
    conv_lstm_hid_depth = opt['conv_lstm_hid_depth']
    wd = opt['weight_decay']

    # Input image, [B, H, W, 3]
    x = tf.placeholder('float', [None, inp_height, inp_width, 3])
    # Groundtruth segmentation maps, [B, T, H, W]
    y_gt = tf.placeholder('float', [None, timespan, inp_height, inp_width])
    # Groundtruth confidence score, [B, T]
    s_gt = tf.placeholder('float', [None, timespan])

    y_gt_list = tf.split(1, timespan, y_gt)

    # 1st convolution layer
    w_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = weight_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd convolution layer
    w_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = weight_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 3rd convolution layer
    w_conv3 = weight_variable([3, 3, 32, 64])
    b_conv3 = weight_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    w_conv4 = weight_variable([3, 3, 16, 1])
    b_conv4 = weight_variable([1])
    b_5 = weight_variable([lstm_inp_height, lstm_inp_width])

    w_6 = weight_variable([inp_width / 32 * inp_height / 32, 1])
    b_6 = weight_variable([1])

    lstm_depth = 16
    lstm_inp_height = inp_height / 8
    lstm_inp_width = inp_width / 8
    c_init = tf.zeros([lstm_inp_height, lstm_inp_width, lstm_depth])
    h_init = tf.zeros([lstm_inp_height, lstm_inp_width, lstm_depth])

    unroll_conv_lstm = _add_conv_lstm(
        model=m,
        timespan=timespan,
        inp_height=lstm_inp_height,
        inp_width=lstm_inp_width,
        inp_depth=64,
        filter_size=conv_lstm_filter_size,
        hid_depth=16,
        c_init=c_init,
        h_init=h_init,
        wd=wd,
        name='lstm'
    )
    h_lstm = m['h_lstm']

    h_conv4 = [None] * timespan
    segm_lo = [None] * timespan
    segm_gt_lo = [None] * timespan
    obj = [None] * timespan

    for t in xrange(timespan):
        unroll_conv_lstm(h_pool3, time=t)

        # Segmentation network
        h_conv4 = tf.nn.relu(conv2d(h_lstm[t], w_conv4) + b_conv4)
        segm_lo[t] = tf.expand_dims(tf.siigmoid(
            tf.log(tf.nn.softmax(h_conv4)) + b_5), dim=1)
        segm_gt_lo[t] = avg_pool_4x4(tf.reshape(
            y_gt_list, [-1, inp_height, inp_width, 1]))

        # Objectness network
        h_pool4 = max_pool_4x4(h_lstm[t])
        obj[t] = tf.expand_dims(tf.sigmoid(
            tf.matmul(hpool4, w_6) + b_6), dim=1)

    # Loss function
    y_out = tf.concat(1, segm_lo)
    y_gt = tf.concat(1, segm_gt_lo)
    s_out = tf.concat(1, obj)
    # Subsample groundtruth

    segm_gt = tf.placeholder('float', [None, out_size, out_size])
    segm_gt_reshape = tf.reshape(y_gt_list, [-1, out_size, out_size, 1])

    if opt['output_downsample'] > 1:
        segm_gt_lo = avg_pool_4x4(segm_gt_reshape)
    else:
        segm_gt_lo = segm_gt_reshape
    y_gt = tf.placeholder('float', [None, None, inp_height, inp_width])

    r = 1.0
    loss = ins_segm_loss(y_out, y_gt, s_out, s_gt, r)
    m['loss'] = loss
    tf.add_to_collection('losses', loss)

    train_step = GradientClipOptimizer(
        tf.train.AdamOptimizer(lr, epsilon=eps), clip=1.0).minimize(
        total_loss)
    m['train_step'] = train_step

    return m


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
    # Number of steps
    kNumSteps = 500000
    # Number of steps per checkpoint
    kStepsPerCkpt = 1000
    parser = argparse.ArgumentParser(
        description='Train DRAW')
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
    parser.add_argument('-seed', default=100, type=int,
                        help='Training seed')
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

    opt = {
    }

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    # dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)

    m = get_model(opt, device=device)
    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())

    task_name = 'draw_mnist'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)
    results_folder = args.results
    logs_folder = args.logs
    exp_folder = os.path.join(results_folder, model_id)
    exp_logs_folder = os.path.join(logs_folder, model_id)

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

    random = np.random.RandomState(args.seed)

    step = 0
    while step < loop_config['num_steps']:
        # Validation
        valid_ce = 0
        log.info('Running validation')
        for ii in xrange(100):
            batch = dataset.test.next_batch(100)
            x = preprocess(batch[0], opt)
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['hid_dim']])
            ce = sess.run(m['ce'], feed_dict={
                m['x']: x,
                m['u']: u
            })
            valid_ce += ce * 100 / 10000.0
        log.info('step {:d}, valid ce {:.4f}'.format(step, valid_ce))
        valid_ce_logger.add(step, valid_ce)

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x = preprocess(batch[0], opt)
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['hid_dim']])
            r = sess.run([m['ce'], m['train_step']], feed_dict={
                m['x']: x,
                m['u']: u
            })
            if step % 10 == 0:
                ce = r[0]
                log.info('{:d} train ce {:.4f} t {:.2f}ms'.format(
                    step, ce, (time.time() - st) * 1000))
                train_ce_logger.add(step, ce)

            step += 1

            # Save model
            if step % args.steps_per_ckpt == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)

    sess.close()
    pass
