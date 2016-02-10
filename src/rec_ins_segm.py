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
from tensorflow.python.framework import ops
from utils import logger
from utils import log_manager
from utils import saver
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
    segm_data = data.get_instance_segmentation_data(opt, image_data)
    dataset['train'] = segm_data

    opt['num_examples'] = num_valid
    raw_data = data.get_raw_data(opt, seed=3)
    image_data = data.get_image_data(opt, raw_data)
    segm_data = data.get_instance_segmentation_data(opt, image_data)
    dataset['valid'] = segm_data

    return dataset


def _get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'CumMin',
                      'CumMinGrad', 'Hungarian', 'Reverse'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise
            # CPU.
            return device

    return _device_fn


def _get_batch_fn(dataset):
    """
    Preprocess mini-batch data given start and end indices.
    """
    def get_batch(start, end):
        x_bat = dataset['input'][start: end]
        y_bat = dataset['label_segmentation'][start: end]
        s_bat = dataset['label_score'][start: end]
        x_bat, y_bat, s_bat = preprocess(x_bat, y_bat, s_bat)

        return x_bat, y_bat, s_bat

    return get_batch


def preprocess(inp, label_segmentation, label_score):
    """Preprocess training data."""
    return (inp.astype('float32') / 255,
            label_segmentation.astype('float32'),
            label_score.astype('float32'))


def _conv2d(x, w):
    """2-D convolution."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def _max_pool_4x4(x):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME')


def _weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _add_conv_lstm(model, timespan, inp_height, inp_width, inp_depth, filter_size, hid_depth, c_init, h_init, wd=None, name=''):
    """Adds a Conv-LSTM component.

    Args:
        model: Model dictionary
        timespan: Maximum length of the LSTM
        inp_height: Input image height
        inp_width: Input image width
        inp_depth: Input image depth
        filter_size: Conv gate filter size
        hid_depth: Hidden state depth
        c_init: Cell state initialization
        h_init: Hidden state initialization
        wd: Weight decay
        name: Prefix
    """
    g_i = [None] * timespan
    g_f = [None] * timespan
    g_o = [None] * timespan
    u = [None] * timespan
    c = [None] * (timespan + 1)
    h = [None] * (timespan + 1)
    c[-1] = c_init
    h[-1] = h_init

    # Input gate
    w_xi = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xi_{}'.format(name))
    w_hi = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hi_{}'.format(name))
    b_i = _weight_variable([hid_depth],
                           wd=wd, name='b_i_{}'.format(name))

    # Forget gate
    w_xf = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xf_{}'.format(name))
    w_hf = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hf_{}'.format(name))
    b_f = _weight_variable([hid_depth],
                           wd=wd, name='b_f_{}'.format(name))

    # Input activation
    w_xu = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xu_{}'.format(name))
    w_hu = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hu_{}'.format(name))
    b_u = _weight_variable([hid_depth],
                           wd=wd, name='b_u_{}'.format(name))

    # Output gate
    w_xo = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xo_{}'.format(name))
    w_ho = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_ho_{}'.format(name))
    b_o = _weight_variable([hid_depth], name='b_o_{}'.format(name))

    def unroll(inp, time):
        t = time
        g_i[t] = tf.sigmoid(_conv2d(inp, w_xi) + _conv2d(h[t - 1], w_hi) + b_i)
        g_f[t] = tf.sigmoid(_conv2d(inp, w_xf) + _conv2d(h[t - 1], w_hf) + b_f)
        g_o[t] = tf.sigmoid(_conv2d(inp, w_xo) + _conv2d(h[t - 1], w_ho) + b_o)
        u[t] = tf.tanh(_conv2d(inp, w_xu) + _conv2d(h[t - 1], w_hu) + b_u)
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


# Register gradient for Hungarian algorithm.
ops.NoGradient("Hungarian")


# Register gradient for cumulative minimum operation.
@ops.RegisterGradient("CumMin")
def _cum_min_grad(op, grad):
    """The gradients for `cum_min`.

    Args:
        op: The `cum_min` `Operation` that we are differentiating, which we can
        use to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `cum_min` op.

    Returns:
        Gradients with respect to the input of `cum_min`.
    """
    x = op.inputs[0]
    return [tf.user_ops.cum_min_grad(grad, x)]


def _f_iou(a, b, timespan, pairwise=False):
    """
    Computes IOU score.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, N, H, W], or [N, H, W], or [H, W]
           in pairwise mode, the second dimension can be different,
           e.g. [B, M, H, W], or [M, H, W], or [H, W]
        pairwise: whether the inputs are already aligned, outputs [B, N] or
                  the inputs are orderless, outputs [B, N, M].
    """
    eps = 1e-5

    def _get_reduction_indices(a):
        """Gets the list of axes to sum over."""
        dim = len(a.get_shape())

        return [dim - 2, dim - 1]

    def _inter(a, b):
        """Computes intersection."""
        reduction_indices = _get_reduction_indices(a)
        return tf.reduce_sum(a * b, reduction_indices=reduction_indices)

    def _union(a, b):
        """Computes union."""
        reduction_indices = _get_reduction_indices(a)
        return tf.reduce_sum(a + b - (a * b) + eps,
                             reduction_indices=reduction_indices)
    if pairwise:
        # b_shape = tf.shape(b)
        # # [1, 1, M, 1, 1]
        # a_shape2 = tf.concat(0, [tf.constant([1, 1]),
        #                          b_shape[1: 2],
        #                          tf.constant([1, 1])])
        # # [B, N, H, W] => [B, N, 1, H, W] => [B, N, M, H, W]
        # a = tf.expand_dims(a, 2)
        # # [B, M, H, W] => [B, 1, M, H, W]
        # b = tf.expand_dims(b, 1)
        # a = tf.tile(a, a_shape2)
        # return _inter(a, b) / _union(a, b)

        y_list = [None] * timespan
        a_list = [None] * timespan
        # [B, N, H, W] => [B, N, 1, H, W]
        a = tf.expand_dims(a, 2)
        # [B, N, 1, H, W] => N * [B, 1, 1, H, W]
        a_list = tf.split(1, timespan, a)
        # [B, M, H, W] => [B, 1, M, H, W]
        b = tf.expand_dims(b, 1)

        for ii in xrange(timespan):
            # [B, 1, M]
            y_list[ii] = _inter(a_list[ii], b) / _union(a_list[ii], b)

        # N * [B, 1, M] => [B, N, M]
        return tf.concat(1, y_list)

    else:
        return _inter(a, b) / _union(a, b)


def _cum_min(s, d):
    """Calculates cumulative minimum.

    Args:
        s: Input matrix [B, D].
        d: Second dim.

    Returns:
        s_min: [B, D], cumulative minimum accross the second dim.
    """
    s_min_list = [None] * d
    s_min_list[0] = s[:, 0: 1]
    for ii in xrange(1, d):
        s_min_list[ii] = tf.minimum(s_min_list[ii - 1], s[:, ii: ii + 1])

    return tf.concat(1, s_min_list)


def _add_ins_segm_loss(model, y_out, y_gt, s_out, s_gt, r, timespan, use_cum_min=True, has_segm=True):
    """
    Instance segmentation loss.

    Args:
        y_out: [B, N, H, W], output segmentations.
        y_gt: [B, M, H, W], groundtruth segmentations.
        s_out: [B, N], output confidence score.
        s_gt: [B. M], groundtruth confidence score.
        r: float, mixing coefficient for combining segmentation loss and
        confidence score loss.
    """

    def _bce(y_out, y_gt):
        """Binary cross entropy."""
        eps = 1e-7
        return -y_gt * tf.log(y_out + eps) - \
            (1 - y_gt) * tf.log(1 - y_out + eps)

    # IOU score, [B, N, M]
    iou_soft = _f_iou(y_out, y_gt, timespan, pairwise=True)
    model['iou_soft'] = iou_soft
    # Matching score, [B, N, M]
    # Add small epsilon because the matching algorithm only accepts complete
    # bipartite graph with positive weights.
    # Mask out the items beyond the total groudntruth count.
    # Mask X, [B, M] => [B, 1, M]
    mask_x = tf.expand_dims(s_gt, dim=1)
    # Mask Y, [B, M] => [B, N, 1]
    mask_y = tf.expand_dims(s_gt, dim=2)
    iou_mask = iou_soft * mask_x * mask_y

    # Keep certain precision so that we can get optimal matching within
    # reasonable time.
    eps = 1e-5
    precision = 1e6
    iou_mask = tf.round(iou_mask * precision) / precision
    match_eps = tf.user_ops.hungarian(iou_mask + eps)[0]

    # [1, N, 1, 1]
    y_out_shape = tf.shape(y_out)
    num_segm_out = y_out_shape[1: 2]
    num_segm_out_mul = tf.concat(
        0, [tf.constant([1]), num_segm_out, tf.constant([1])])
    # Mask the graph algorithm output.
    match = match_eps * mask_x * mask_y
    model['match'] = match

    # Loss for confidence scores.
    if use_cum_min:
        # [B, N]
        s_out_min = _cum_min(s_out, timespan)
        # [B, N, M] => [B, N]
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        # [B, N]
        s_bce = _bce(s_out_min, match_sum)
        model['s_out_min'] = s_out_min
    else:
        # Try simply do binary xent for confidence sequence.
        s_bce = _bce(s_out, s_gt)
    model['s_bce'] = s_bce

    # Loss normalized by number of examples.
    y_gt_shape = tf.shape(y_gt)
    num_ex = tf.to_float(y_gt_shape[0])
    max_num_obj = tf.to_float(y_gt_shape[1])

    # [B, N, M] => scalar
    conf_loss = r * tf.reduce_sum(s_bce) / num_ex / max_num_obj
    segm_loss = -tf.reduce_sum(iou_soft * match) / num_ex / max_num_obj
    model['segm_loss'] = segm_loss
    if has_segm:
        loss = segm_loss + conf_loss
    else:
        loss = conf_loss

    model['conf_loss'] = conf_loss
    model['loss'] = loss

    # Counting accuracy
    count_out = tf.reduce_sum(tf.to_float(s_out > 0.5), reduction_indices=[1])
    count_gt = tf.reduce_sum(s_gt, reduction_indices=[1])
    count_acc = tf.reduce_sum(tf.to_float(count_out == count_gt)) / num_ex
    model['count_acc'] = count_acc

    # Hard IOU
    iou_hard = _f_iou(tf.to_float(y_out > 0.5), y_gt, timespan, pairwise=True)
    iou_hard = iou_hard * match / num_ex / max_num_obj
    model['iou_hard'] = iou_hard

    return loss


def get_model(opt, device='/cpu:0', train=True):
    """Get model."""
    model = {}
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    conv_lstm_filter_size = opt['conv_lstm_filter_size']
    conv_lstm_hid_depth = opt['conv_lstm_hid_depth']
    wd = opt['weight_decay']
    has_segm = ('has_segm' not in opt) or opt['has_segm']

    with tf.device(_get_device_fn(device)):
        # Input image, [B, H, W, 3]
        x = tf.placeholder('float', [None, inp_height, inp_width, 3])
        # Groundtruth segmentation maps, [B, T, H, W]
        y_gt = tf.placeholder('float', [None, timespan, inp_height, inp_width])
        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])
        y_gt_list = tf.split(1, timespan, y_gt)
        model['x'] = x
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt

        # Possibly add random image transformation layers here in training time.
        # Need to combine x and y together to crop.
        # Other operations on x only.
        # x = tf.image.random_crop()
        # x = tf.image.random_flip()

        # 1st convolution layer
        # [B, H, W, 3] => [B, H / 2, W / 2, 16]
        w_conv1 = _weight_variable([3, 3, 3, 16])
        b_conv1 = _weight_variable([16])
        h_conv1 = tf.nn.relu(_conv2d(x, w_conv1) + b_conv1)
        h_pool1 = _max_pool_2x2(h_conv1)

        # 2nd convolution layer
        # [B, H / 2, W / 2, 16] => [B, H / 4, W / 4, 32]
        w_conv2 = _weight_variable([3, 3, 16, 32])
        b_conv2 = _weight_variable([32])
        h_conv2 = tf.nn.relu(_conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = _max_pool_2x2(h_conv2)

        # 3rd convolution layer
        # [B, H / 4, W / 4, 32] => [B, H / 8, W / 8, 64]
        w_conv3 = _weight_variable([3, 3, 32, 64])
        b_conv3 = _weight_variable([64])
        h_conv3 = tf.nn.relu(_conv2d(h_pool2, w_conv3) + b_conv3)
        h_pool3 = _max_pool_2x2(h_conv3)

        lstm_depth = 16
        lstm_height = inp_height / 8
        lstm_width = inp_width / 8

        # ConvLSTM hidden state initialization
        # [B, LH, LW, LD]
        x_shape = tf.shape(x)
        num_ex = x_shape[0: 1]
        c_init = tf.zeros(tf.concat(
            0, [num_ex, tf.constant([lstm_height, lstm_width, lstm_depth])]))
        h_init = tf.zeros(tf.concat(
            0, [num_ex, tf.constant([lstm_height, lstm_width, lstm_depth])]))

        # Segmentation network
        # 4th convolution layer (on ConvLSTM output).
        w_conv4 = _weight_variable([3, 3, 16, 1])
        b_conv4 = _weight_variable([1])

        # Bias towards segmentation output.
        b_5 = _weight_variable([lstm_height * lstm_width])

        # Confidence network
        # Linear layer for output confidence score.
        w_6 = _weight_variable(
            [lstm_height * lstm_height / 16 * lstm_depth, 1])
        b_6 = _weight_variable([1])

        unroll_conv_lstm = _add_conv_lstm(
            model=model,
            timespan=timespan,
            inp_height=lstm_height,
            inp_width=lstm_width,
            inp_depth=64,
            filter_size=conv_lstm_filter_size,
            hid_depth=16,
            c_init=c_init,
            h_init=h_init,
            wd=wd,
            name='lstm'
        )
        h_lstm = model['h_lstm']

        h_conv4 = [None] * timespan
        segm_lo = [None] * timespan
        # segm_out = [None] * timespan
        score = [None] * timespan
        h_pool4 = [None] * timespan

        for t in xrange(timespan):
            # We can potentially have another canvas that substract this one.
            unroll_conv_lstm(h_pool3, time=t)

            # Segmentation network
            # [B, LH, LW, 1]
            h_conv4 = tf.nn.relu(_conv2d(h_lstm[t], w_conv4) + b_conv4)
            # [B, LH * LW]
            h_conv4_reshape = tf.reshape(
                h_conv4, [-1, lstm_height * lstm_width])
            # [B, LH * LW] => [B, LH, LW] => [B, 1, LH, LW]
            # [B, LH * LW] => [B, LH, LW] => [B, LH, LW, 1]
            segm_lo[t] = tf.expand_dims(tf.reshape(tf.sigmoid(
                tf.log(tf.nn.softmax(h_conv4_reshape)) + b_5),
                [-1, lstm_height, lstm_width]), dim=3)

            # Objectness network
            # [B, LH, LW, LD] => [B, LLH, LLW, LD] => [B, LLH * LLW * LD]
            h_pool4[t] = tf.reshape(_max_pool_4x4(h_lstm[t]),
                                    [-1,
                                     lstm_height * lstm_width / 16 * lstm_depth])
            # [B, LLH * LLW * LD] => [B, 1]
            score[t] = tf.sigmoid(tf.matmul(h_pool4[t], w_6) + b_6)

        # [B * T, LH, LW, 1]
        segm_lo_all = tf.concat(0, segm_lo)

        # [B * T, LH, LW, 1] => [B * T, H, W, 1] => [B, T, H, W]
        y_out = tf.reshape(
            tf.image.resize_bilinear(segm_lo_all, [inp_height, inp_width]),
            [-1, timespan, inp_height, inp_width])

        model['y_out'] = y_out

        # T * [B, 1] = [B, T]
        s_out = tf.concat(1, score)
        model['s_out'] = s_out

        model['h_lstm_0'] = h_lstm[0]
        model['h_pool4_0'] = h_pool4[0]
        model['s_0'] = score[0]

        # Loss function
        if train:
            r = opt['loss_mix_ratio']
            lr = opt['learning_rate']
            use_cum_min = ('cum_min' not in opt) or opt['cum_min']
            eps = 1e-7
            loss = _add_ins_segm_loss(
                model, y_out, y_gt, s_out, s_gt, r, timespan,
                use_cum_min=use_cum_min, has_segm=has_segm)
            tf.add_to_collection('losses', loss)
            total_loss = tf.add_n(tf.get_collection(
                'losses'), name='total_loss')
            model['total_loss'] = total_loss

            train_step = GradientClipOptimizer(
                tf.train.AdamOptimizer(lr, epsilon=eps),
                clip=1.0).minimize(total_loss)
            # train_step = tf.train.GradientDescentOptimizer(
            #         lr).minimize(total_loss)
            model['train_step'] = train_step

    return model


def _parse_args():
    """Parse input arguments."""
    # Default dataset options
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
    kNumExamples = 200
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

    # Default model options
    kWeightDecay = 5e-5
    kLearningRate = 1e-3
    kLossMixRatio = 1.0
    kConvLstmFilterSize = 5
    kConvLstmHiddenDepth = 64

    # Default training options
    # Number of steps
    kNumSteps = 500000
    # Number of steps per checkpoint
    kStepsPerCkpt = 1000

    parser = argparse.ArgumentParser(
        description='Train DRAW')

    # Dataset options
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

    # Model options
    parser.add_argument('-weight_decay', default=kWeightDecay, type=float,
                        help='Weight L2 regularization')
    parser.add_argument('-learning_rate', default=kLearningRate, type=float,
                        help='Model learning rate')
    parser.add_argument('-loss_mix_ratio', default=kLossMixRatio, type=float,
                        help='Mix ratio between segmentation and score loss')
    parser.add_argument('-conv_lstm_filter_size', default=kConvLstmFilterSize,
                        type=int, help='Conv LSTM filter size')
    parser.add_argument('-conv_lstm_hid_depth', default=kConvLstmHiddenDepth,
                        type=int, help='Conv LSTM hidden depth')

    # Test arguments.
    # To see the effect of cumulative minimum.
    parser.add_argument('-no_cum_min', action='store_true',
                        help='Whether cumulative minimum. Default yes.')

    # Test arguments.
    # Only to use when only care about the count.
    # Still segment images, the segmentation loss does not get back propagated.
    parser.add_argument('-no_segm', action='store_true',
                        help='Whether has segmentation network.')

    # Training options
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
    parser.add_argument('-restore', default=None,
                        help='Model save folder to restore from')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)

    # Logistics
    task_name = 'rec_ins_segm'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)
    results_folder = args.results
    logs_folder = args.logs
    exp_folder = os.path.join(results_folder, model_id)
    exp_logs_folder = os.path.join(logs_folder, model_id)

    # Logger
    log = logger.get(os.path.join(exp_logs_folder, 'raw'))
    log.log_args()

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    # Train loop options
    train_opt = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    # Restore previously saved checkpoints.
    if args.restore:
        ckpt_info = saver.get_ckpt_info(args.restore)
        model_opt = ckpt_info['model_opt']
        data_opt = ckpt_info['data_opt']
        ckpt_fname = ckpt_info['ckpt_fname']
        step = ckpt_info['step']
    else:
        log.info('Initializing new model')
        model_opt = {
            'inp_height': args.height,
            'inp_width': args.width,
            'timespan': args.max_num_objects,
            'weight_decay': args.weight_decay,
            'learning_rate': args.learning_rate,
            'loss_mix_ratio': args.loss_mix_ratio,
            'conv_lstm_filter_size': args.conv_lstm_filter_size,
            'conv_lstm_hid_depth': args.conv_lstm_hid_depth,
            'cum_min': not args.no_cum_min,
            'has_segm': not args.no_segm
        }
        data_opt = {
            'height': args.height,
            'width': args.width,
            'radius_upper': args.radius_upper,
            'radius_lower': args.radius_lower,
            'border_thickness': args.border_thickness,
            'max_num_objects': args.max_num_objects,
            'num_object_types': args.num_object_types,
            'center_var': args.center_var,
            'size_var': args.size_var
        }
        step = 0

    dataset = get_dataset(data_opt, args.num_ex, args.num_ex / 10)
    m = get_model(model_opt, device=device)
    # sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if args.restore:
        saver.restore_ckpt(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series logger
    if args.logs:
        train_loss_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'train_loss.csv'), 'train loss',
            name='Training loss',
            buffer_size=10)
        valid_loss_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'valid_loss.csv'), 'valid loss',
            name='Validation loss',
            buffer_size=1)
        valid_iou_hard_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'valid_iou_hard.csv'), 'valid iou',
            name='Validation IoU hard',
            buffer_size=1)
        valid_iou_soft_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'valid_iou_soft.csv'), 'valid iou',
            name='Validation IoU soft',
            buffer_size=1)
        valid_count_acc_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'valid_count_acc.csv'),
            'valid count acc',
            name='Validation count accuracy',
            buffer_size=1)
        step_time_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'step_time.csv'), 'step time (ms)',
            name='Step time',
            buffer_size=10)
        log_manager.register(log.filename, 'plain', 'Raw logs')
        log.info(
            'Visualization can be viewed at: http://{}/visualizer?id={}'.format(
                args.localhost, model_id))

    num_ex_train = dataset['train']['input'].shape[0]
    num_ex_valid = dataset['valid']['input'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    get_batch_valid = _get_batch_fn(dataset['valid'])
    batch_size_train = 32
    batch_size_valid = 100
    log.info('Number of validation examples: {}'.format(num_ex_valid))
    log.info('Validation batch size: {}'.format(batch_size_valid))
    log.info('Number of training examples: {}'.format(num_ex_train))
    log.info('Training batch size: {}'.format(batch_size_train))

    # Train loop
    while step < train_opt['num_steps']:
        # Validation
        loss = 0.0
        hard_iou = 0.0
        soft_iou = 0.0
        count_acc = 0.0
        segm_loss = 0.0
        conf_loss = 0.0
        log.info('Running validation')
        for x_bat, y_bat, s_bat in BatchIterator(num_ex_valid,
                                                 batch_size=batch_size_valid,
                                                 get_fn=get_batch_valid,
                                                 progress_bar=False):
            r = sess.run([m['loss'], m['segm_loss'], m['conf_loss'],
                          m['iou_soft'], m['iou_hard'], m['count_acc']],
                         feed_dict={
                m['x']: x_bat,
                m['y_gt']: y_bat,
                m['s_gt']: s_bat
            })
            _loss = r[0]
            _segm_loss = r[1]
            _conf_loss = r[2]
            _iou_soft = r[3]
            _iou_hard = r[4]
            _count_acc = r[5]
            loss += _loss * batch_size_valid / float(num_ex_valid)
            segm_loss += _segm_loss * batch_size_valid / float(num_ex_valid)
            conf_loss += _conf_loss * batch_size_valid / float(num_ex_valid)
            iou_soft += _iou_soft * batch_size_valid / float(num_ex_valid)
            iou_hard += _iou_hard * batch_size_valid / float(num_ex_valid)
            count_acc += _count_acc * batch_size_valid / float(num_ex_valid)

        log.info(('{:d} valid loss {:.4f} segm_loss {:.4f} conf_loss {:.4f} '
                  'iou soft {:.4f} iou hard {:.4f} count acc {:.4f}').format(
            step, loss, segm_loss, conf_loss, iou_soft, iou_hard, count_acc))
        valid_loss_logger.add(step, valid_loss)
        valid_iou_soft_logger(step, iou_soft)
        valid_iou_hard_logger(ste, iou_hard)
        valid_count_acc_logger.add(step, count_acc)

        # Train
        for x_bat, y_bat, s_bat in BatchIterator(num_ex_train,
                                                 batch_size=batch_size_train,
                                                 get_fn=get_batch_train,
                                                 progress_bar=False):
            start_time = time.time()
            r = sess.run([m['loss'], m['train_step']], feed_dict={
                m['x']: x_bat,
                m['y_gt']: y_bat,
                m['s_gt']: s_bat
            })

            # Print statistics
            if step % 10 == 0:
                step_time = (time.time() - start_time) * 1000
                loss = r[0]
                log.info('{:d} train loss {:.4f} t {:.2f}ms'.format(
                    step, loss, step_time))
                train_loss_logger.add(step, loss)
                step_time_logger.add(step, step_time)

            # Save model
            if step % train_opt['steps_per_ckpt'] == 0:
                saver.save_ckpt(exp_folder, sess, model_opt,
                                data_opt, global_step=step)

            step += 1

    sess.close()
    train_loss_logger.close()
    valid_loss_logger.close()
    valid_iou_soft_logger.close()
    valid_iou_hard_logger.close()
    valid_count_acc_logger.close()
    step_time_logger.close()
    pass
