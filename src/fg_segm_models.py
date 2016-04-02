import cslab_environ

import tensorflow as tf

from ris_base import *
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import nnlib as nn
import image_ops as img

log = logger.get()


def get_model(opt, device='/cpu:0'):
    """A fully-convolutional neural network for foreground segmentation."""
    model = {}
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']
    cnn_filter_size = opt['cnn_filter_size']
    cnn_depth = opt['cnn_depth']
    cnn_pool = opt['cnn_pool']
    dcnn_filter_size = opt['dcnn_filter_size']
    dcnn_depth = opt['dcnn_depth']
    dcnn_pool = opt['dcnn_pool']
    use_bn = opt['use_bn']
    wd = opt['weight_decay']
    rnd_hflip = opt['rnd_hflip']
    rnd_vflip = opt['rnd_vflip']
    rnd_transpose = opt['rnd_transpose']
    rnd_colour = opt['rnd_colour']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
    add_skip_conn = opt['add_skip_conn']

    with tf.device(get_device_fn(device)):
        x = tf.placeholder('float', [None, inp_height, inp_width, inp_depth])
        y_gt = tf.placeholder('float', [None, inp_height, inp_width, 1])
        phase_train = tf.placeholder('bool')
        model['x'] = x
        model['y_gt'] = y_gt
        model['phase_train'] = phase_train

        global_step = tf.Variable(0.0)
        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        x, y_gt = img.random_transformation2(
            x, y_gt, padding, phase_train,
            rnd_hflip=rnd_hflip, rnd_vflip=rnd_vflip,
            rnd_transpose=rnd_transpose, rnd_colour=rnd_colour)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

        cnn_nlayers = len(cnn_filter_size)
        cnn_channels = [inp_depth] + cnn_depth
        cnn_act = [tf.nn.relu] * cnn_nlayers
        cnn_use_bn = [use_bn] * cnn_nlayers
        cnn = nn.cnn(cnn_filter_size, cnn_channels, cnn_pool, cnn_act,
                     cnn_use_bn, phase_train=phase_train, wd=wd, model=model)
        h_cnn = cnn(x)

        dcnn_nlayers = len(dcnn_filter_size)
        dcnn_act = [tf.nn.relu] * (dcnn_nlayers - 1) + [None]
        if add_skip_conn:
            dcnn_skip_ch = [0] + cnn_channels[::-1][1:] + [inp_depth]
            dcnn_skip = [None] + h_cnn[::-1][1:] + [x]
        else:
            dcnn_skip_ch = None
            dcnn_skip = None
        dcnn_channels = [cnn_channels[-1]] + dcnn_depth
        dcnn_use_bn = [use_bn] * dcnn_nlayers
        dcnn = nn.dcnn(dcnn_filter_size, dcnn_channels, dcnn_pool, dcnn_act,
                       dcnn_use_bn, skip_ch=dcnn_skip_ch,
                       phase_train=phase_train, wd=wd)
        h_dcnn = dcnn(h_cnn[-1], skip=dcnn_skip)

        y_out = tf.reshape(h_dcnn[-1], [-1, inp_height, inp_width])
        y_out = tf.sigmoid(y_out)
        model['y_out'] = y_out

        num_ex = tf.to_float(num_ex)
        _y_gt = tf.reshape(y_gt, [-1, inp_height, inp_width])
        iou_soft = tf.reduce_sum(f_iou(y_out, _y_gt)) / num_ex
        model['iou_soft'] = iou_soft
        iou_hard = tf.reduce_sum(
            f_iou(tf.to_float(y_out > 0.5), _y_gt)) / num_ex
        model['iou_hard'] = iou_hard
        loss = -iou_soft
        model['loss'] = -iou_soft
        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        eps = 1e-7
        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

    return model
