"""
This code implements DRAW (Deep Recurrent Attention Writer) [2] on MNIST.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:

Reference:
[1] K. Gregor, I. Danihelka, A. Grabes, D.J. Rezende, D. Wierstra. DRAW: A
Recurrent Neural Network For Image Generation. ICML 2015.
"""

import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.5.0/lib/python2.7/site-packages')

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

log = logger.get()

# Number of steps
kNumSteps = 500000
# Number of steps per checkpoint
kStepsPerCkpt = 1000


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_train_model(opt, device='/cpu:0'):
    """Get train model for DRAW."""

    #########################################################
    #########################################################
    # Constants
    #########################################################
    #########################################################
    # RNN timespan: T.
    timespan = opt['timespan']

    # Input image height: H.
    inp_height = opt['inp_height']
    # Input image width: W.
    inp_width = opt['inp_width']
    # Gaussian filter size: F.
    filter_size = opt['filter_size']

    # Number of hidden units in the RNN encoder: He.
    num_hid_enc = opt['num_hid_enc']
    # Number of hidden units in the RNN decoder: Hd.
    num_hid_dec = opt['num_hid_dec']
    # Number of hidden dimension in the latent variable: Hz.
    num_hid = opt['num_hid']

    # Weight L2 regularization.
    wd = opt['weight_decay']

    # Constant for computing filter_x. Shape: [1, W, 1].
    span_x = np.reshape(np.arange(inp_width), [1, inp_width, 1])
    # Constant for computing filter_y. Shape: [1, H, 1].
    span_y = np.reshape(np.arange(inp_height), [1, inp_height, 1])

    #########################################################
    #########################################################
    # Variables
    #########################################################
    #########################################################
    with tf.device(device):
        # Input image. Shape: [B, T, H, W].
        x = tf.placeholder('float', [None, inp_width, inp_height])
        # [B, H * W]
        x_flat = tf.reshape(x, [-1, inp_height * inp_width], name='x_flat')
        # Noise. Shape: [B, T, Hz]
        u = tf.placeholder('float', [None, timespan, num_hid])
        u_l = tf.split(1, timespan, u)
        # Error images (original substracted by drawn). Shape: T * [B, H, W].
        x_err = [None] * timespan
        x_err_flat = [None] * timespan
        u_l_flat = [None] * timespan

        # Control variale. [g_x, g_y, log_var, log_delta, log_gamma].
        # Shape: T * [B, 5].
        ctl = [None] * timespan
        # Attention center x. Shape: T * [B].
        ctr_x = [None] * timespan
        # Attention center y. Shape: T * [B].
        ctr_y = [None] * timespan
        # Attention width. Shape: T * [B].
        delta = [None] * timespan
        # Log of the variance of Gaussian filter. Shape: T * [B].
        lg_var = [None] * timespan
        # Log of the multiplier of Gaussian filter. Shape: T * [B].
        lg_gamma = [None] * timespan

        # Gaussian filter mu_x. Shape: T * [B, W, F].
        mu_x = [None] * timespan
        # Gaussian filter mu_y. Shape: T * [B, H, F].
        mu_y = [None] * timespan
        # Gaussian filter on x direction. Shape: T * [B, W, F].
        filter_x = [None] * timespan
        filter_x_t = [None] * timespan
        # Gaussian filter on y direction. Shape: T * [B, H, F].
        filter_y = [None] * timespan
        filter_y_t = [None] * timespan

        # Read out image from the read head. Shape: T * [B, F, F].
        readout_x = [None] * timespan
        readout_err = [None] * timespan
        readout = [None] * timespan

        # Hidden state of the RNN encoder. Shape: (T + 1) * [B, He].
        h_enc = [None] * (timespan + 1)
        # Input gate of the RNN encoder. Shape: T * [B, He].
        gi_enc = [None] * timespan
        # Recurrent gate of the RNN encoder. Shape: T * [B, He].
        gr_enc = [None] * timespan
        # Hidden candidate activation of the RNN encoder.
        h_cdd_enc = [None] * timespan

        # Mean of hidden latent variable. Shape: T * [B, Hz].
        mu_z = [None] * timespan
        # Standard deviation of hidden latent variable. Shape: T * [B, Hz].
        sigma_z = [None] * timespan
        # Hidden latent variable. Shape: T * [B, Hz].
        z = [None] * timespan

        # Hidden state of the RNN decoder. Shape: (T + 1) * [B, Hd].
        h_dec = [None] * (timespan + 1)
        # Input gate of the RNN decoder. Shape: T * [B, Hd].
        gi_dec = [None] * timespan
        # Recurrent gate of the RNN decoder. Shape: T * [B, Hd].
        gr_dec = [None] * timespan
        # Hidden candidate activation of the RNN decoder. Shape: T * [B, Hd].
        h_cdd_dec = [None] * timespan

        # Write out. Shape: T * [B, F, F].
        writeout = [None] * timespan

        # Add to canvas. Shape: T * [B, H, W].
        canvas_delta = [None] * timespan
        # Canvas accumulating output. Shape: T * [B, H, W].
        canvas = [None] * (timespan + 1)

        #########################################################
        #########################################################
        # Weights
        #########################################################
        #########################################################
        # From input image to controller variables. Shape: [H * W, 5].
        w_x_ctl = weight_variable([inp_height * inp_width, 5], wd=wd,
                                  name='w_x_ctl')
        # From err image to controller variables. Shape: [H * W, 5].
        w_err_ctl = weight_variable([inp_height * inp_width, 5], wd=wd,
                                    name='w_err_ctl')
        # From hidden decoder to controller variables. Shape: [Hd, 5].
        w_hdec_ctl = weight_variable(
            [num_hid_dec, 5], wd=wd, name='w_hdec_ctl')
        # Controller variable bias. Shape: [5].
        b_ctl = weight_variable([5], wd=wd, name='b_ctl')

        #########################################################
        # Encoder RNN
        #########################################################
        # From read out to input gate encoder. Shape: [2 * F * F, He].
        w_readout_gi_enc = weight_variable(
            [2 * filter_size * filter_size, num_hid_enc], wd=wd,
            name='w_readout_gi_enc')
        # From hidden encoder to input gate encoder. Shape: [He, He].
        w_henc_gi_enc = weight_variable([num_hid_enc, num_hid_enc], wd=wd,
                                        name='w_henc_gi_enc')
        # From hidden decoder to input gate encoder. Shape: [He, He].
        w_hdec_gi_enc = weight_variable([num_hid_dec, num_hid_enc], wd=wd,
                                        name='w_hdec_gi_enc')
        # Input gate encoder bias. Shape: [He].
        b_gi_enc = weight_variable([num_hid_enc], wd=wd, name='b_gi_enc')

        # From read out to recurrent gate encoder. Shape: [2 * F * F, He].
        w_readout_gr_enc = weight_variable(
            [2 * filter_size * filter_size, num_hid_enc], wd=wd,
            name='w_readout_gr_enc')
        # From hidden encoder to recurrent gate encoder. Shape: [He, He].
        w_henc_gr_enc = weight_variable([num_hid_enc, num_hid_enc], wd=wd,
                                        name='w_henc_gr_enc')
        # From hidden decoder to recurrent gate encoder. Shape: [He, He].
        w_hdec_gr_enc = weight_variable(
            [num_hid_dec, num_hid_enc], wd=wd,
            name='w_hdec_gr_enc')
        # Recurrent gate encoder bias. Shape: [He].
        b_gr_enc = weight_variable([num_hid_enc], wd=wd, name='b_gr_enc')

        # From read out to hidden candidate encoder. Shape: [2 * F * F, He].
        w_readout_hcdd_enc = weight_variable(
            [2 * filter_size * filter_size, num_hid_enc], wd=wd,
            name='w_readout_hcdd_enc')
        # From hidden encoder to hidden candidate encoder. Shape: [He, He].
        w_henc_hcdd_enc = weight_variable([num_hid_enc, num_hid_enc], wd=wd,
                                          name='w_henc_hcdd_enc')
        # From hidden decoder to hidden candidate encoder. Shape: [He, He].
        w_hdec_hcdd_enc = weight_variable([num_hid_dec, num_hid_enc], wd=wd,
                                          name='w_hdec_gr_enc')
        # Hidden candidate encoder bias. Shape: [He].
        b_hcdd_enc = weight_variable([num_hid_enc], wd=wd, name='b_hcdd_enc')

        #########################################################
        # Latent distribution
        #########################################################
        # Hidden encoder to latent variable mean.
        w_henc_muz = weight_variable([num_hid_enc, num_hid], wd=wd,
                                     name='w_henc_muz')
        b_muz = weight_variable([num_hid], wd=wd, name='w_henc_muz')
        # Hidden encoder to latent variable std.
        w_henc_stdz = weight_variable(
            [num_hid_enc, num_hid], wd=wd, name='w_henc_stdz')
        b_stdz = weight_variable([num_hid], wd=wd, name='w_henc_stdz')

        #########################################################
        # Decoder RNN
        #########################################################
        # From latent variable to input gate decoder. Shape: [Hz, Hd].
        w_z_gi_dec = weight_variable([num_hid, num_hid_dec], wd=wd,
                                     name='w_z_gi_dec')
        # From hidden decoder to input gate decoder. Shape: [Hd, Hd].
        w_hdec_gi_dec = weight_variable([num_hid_dec, num_hid_dec], wd=wd,
                                        name='w_hdec_gi_dec')
        # Input gate decoder bias. Shape: [Hd].
        b_gi_dec = weight_variable([num_hid_dec], wd=wd, name='b_gi_dec')

        # From latent variable to recurrent gate decoder. Shape: [Hz, Hd].
        w_z_gr_dec = weight_variable([num_hid, num_hid_dec], wd=wd,
                                     name='w_z_gr_dec')
        # From hidden decoder to recurrent gate decoder. Shape: [Hd, Hd].
        w_hdec_gr_dec = weight_variable([num_hid_dec, num_hid_dec], wd=wd,
                                        name='w_hdec_gr_dec')
        # Recurrent gate decoder bias. Shape: [Hd].
        b_gr_dec = weight_variable([num_hid_dec], wd=wd, name='b_gr_dec')

        # From read out to hidden candidate decoder. Shape: [Hz, Hd].
        w_z_hcdd_dec = weight_variable([num_hid, num_hid_dec], wd=wd,
                                       name='w_z_hcdd_dec')
        # From hidden decoder to hidden candidate decoder. Shape: [Hd, Hd].
        w_hdec_hcdd_dec = weight_variable([num_hid_dec, num_hid_dec], wd=wd,
                                          name='w_hdec_hcdd_dec')
        # Hidden candidate decoder bias. Shape: [Hde].
        b_hcdd_dec = weight_variable([num_hid_dec], wd=wd, name='b_hcdd_dec')

        # From decoder to write. Shape: [Hd, F * F].
        w_hdec_writeout = weight_variable(
            [num_hid_dec, filter_size * filter_size], wd=wd,
            name='w_hdec_writeout')
        # Shape: [F * F].
        b_hdec_writeout = weight_variable(
            [filter_size * filter_size], wd=wd, name='b_hdec_writeout')

        # Setup default hidden states.
        x_shape = tf.shape(x, name='x_shape')
        num_ex = x_shape[0: 1]
        num_ex_mul = tf.concat(0, [num_ex, tf.constant([1])])
        h_enc[-1] = tf.tile(tf.zeros([1, num_hid_enc], dtype='float32'),
                            num_ex_mul, name='h_enc_-1')
        h_dec[-1] = tf.tile(tf.zeros([1, num_hid_dec], dtype='float32'),
                            num_ex_mul, name='h_dec_-1')
        canvas[-1] = tf.constant(np.zeros([inp_width, inp_height]),
                                 dtype='float32')

    for t in xrange(timespan):

        with tf.device(device):
            # [B, H * W]
            x_err[t] = tf.sub(x, tf.sigmoid(canvas[t - 1]),
                              name='x_err_{}'.format(t))
            x_err_flat[t] = tf.reshape(x_err[t],
                                       [-1, inp_height * inp_width],
                                       name='x_err_flat_{}'.format(t))
            u_l_flat[t] = tf.reshape(u_l[t], [-1, num_hid],
                                     name='u_flat_{}'.format(t))

            #########################################################
            # Attention controller
            #########################################################
            # [B, 5]
            ctl[t] = tf.add(tf.matmul(x_flat, w_x_ctl) +
                            tf.matmul(x_err_flat[t], w_err_ctl) +
                            tf.matmul(h_dec[t - 1], w_hdec_ctl),
                            b_ctl, name='ctl_{}'.format(t))
            # [B, 1, 1]
            ctr_x[t] = tf.reshape((inp_width + 1) / 2.0 * (ctl[t][:, 0] + 1),
                                  [-1, 1, 1],
                                  name='ctr_x_{}'.format(t))
            # [B, 1, 1]
            ctr_y[t] = tf.reshape((inp_height + 1) / 2.0 * ctl[t][:, 1] + 1,
                                  [-1, 1, 1],
                                  name='ctr_y_{}'.format(t))
            # [B, 1, 1]
            delta[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                  ((filter_size - 1) * tf.exp(ctl[t][:, 3])),
                                  [-1, 1, 1], name='delta_{}'.format(t))
            # [B, 1, 1]
            lg_var[t] = tf.reshape(ctl[t][:, 2], [-1, 1, 1],
                                   name='lg_var_{}'.format(t))
            # [B, 1, 1]
            lg_gamma[t] = tf.reshape(ctl[t][:, 4], [-1, 1, 1],
                                     name='lg_gamma_{}'.format(t))

            #########################################################
            # Gaussian filter
            #########################################################
            # [B, 1, 1] + [B, 1, 1] * [1, F, 1] = [B, 1, F]
            mu_x[t] = tf.add(ctr_x[t], delta[t] * (
                np.reshape(np.arange(filter_size), [1, 1, filter_size]) -
                filter_size / 2.0 - 0.5), name='mu_x_{}'.format(t))
            # [B, 1, 1] + [B, 1, 1] * [1, 1, F] = [B, 1, F]
            mu_y[t] = tf.add(ctr_y[t], delta[t] * (
                np.reshape(np.arange(filter_size), [1, 1, filter_size]) -
                filter_size / 2.0 - 0.5), name='mu_y_{}'.format(t))

            # [B, 1, 1] * [1, W, 1] - [B, 1, F] = [B, W, F]
            filter_x[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_x - mu_x[t]) * (span_x - mu_x[t]) /
                       tf.exp(lg_var[t])),
                name='filter_x_{}'.format(t))
            # [B, F, W]
            filter_x_t[t] = tf.transpose(filter_x[t], [0, 2, 1],
                                         name='filter_x_t_{}'.format(t))
            # [1, H, 1] - [B, 1, F] = [B, H, F]
            filter_y[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_y - mu_y[t]) * (span_y - mu_y[t]) /
                       tf.exp(lg_var[t])),
                name='filter_y_{}'.format(t))
            # [B, F, H]
            filter_y_t[t] = tf.transpose(filter_y[t], [0, 2, 1],
                                         name='filter_y_t_{}'.format(t))

        #########################################################
        # Attention selector
        #########################################################
        # [B, 1, 1] * [B, F, H] * [B, H, W] * [B, W, F] = [B, F, F]
        readout_x[t] = tf.mul(tf.exp(lg_gamma[t]), tf.batch_matmul(
            tf.batch_matmul(filter_y_t[t], x), filter_x[t]),
            name='readout_x_{}'.format(t))
        readout_err[t] = tf.mul(tf.exp(lg_gamma[t]), tf.batch_matmul(
            tf.batch_matmul(filter_y_t[t], x_err[t]), filter_x[t]),
            name='readout_err_{}'.format(t))

        with tf.device(device):
            # [B, 2 * F]
            readout[t] = tf.concat(1,
                                   [tf.reshape(readout_x[t], [-1, filter_size * filter_size]),
                                    tf.reshape(readout_err[t], [-1, filter_size * filter_size])],
                                   name='readout_{}'.format(t))

            #########################################################
            # Encoder RNN
            #########################################################
            # [B, He]
            gi_enc[t] = tf.sigmoid(tf.matmul(readout[t], w_readout_gi_enc) +
                                   tf.matmul(h_enc[t - 1], w_henc_gi_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gi_enc) +
                                   b_gi_enc,
                                   name='gi_enc_{}'.format(t))
            # [B, He]
            gr_enc[t] = tf.sigmoid(tf.matmul(readout[t], w_readout_gr_enc) +
                                   tf.matmul(h_enc[t - 1], w_henc_gr_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gr_enc) +
                                   b_gr_enc,
                                   name='gr_enc_{}'.format(t))
            # [B, He]
            h_cdd_enc[t] = tf.tanh(tf.matmul(readout[t], w_readout_hcdd_enc) +
                                   tf.matmul(h_enc[t - 1], w_henc_hcdd_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_hcdd_enc) +
                                   b_hcdd_enc,
                                   name='h_cdd_enc_{}'.format(t))
            # [B, He]
            h_enc[t] = tf.add(h_enc[t - 1] * (1 - gi_enc[t]),
                              h_cdd_enc[t] * gi_enc[t],
                              name='h_enc_{}'.format(t))

            #########################################################
            # Latent distribution
            #########################################################
            # [B, He] * [He, ]
            mu_z[t] = tf.add(tf.matmul(h_enc[t], w_henc_muz), b_muz,
                             name='mu_z_{}'.format(t))
            sigma_z[t] = tf.exp(tf.matmul(h_enc[t], w_henc_stdz) + b_muz,
                                name='sigma_z_{}'.format(t))
            z[t] = tf.add(mu_z[t], sigma_z[t] * u_l_flat[t],
                          name='z_{}'.format(t))

            #########################################################
            # Decoder RNN
            #########################################################
            # [B, Hd]
            gi_dec[t] = tf.sigmoid(tf.matmul(z[t], w_z_gi_dec) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gi_dec) +
                                   b_gi_dec,
                                   name='gi_dec_{}'.format(t))
            # [B, Hd]
            gr_dec[t] = tf.sigmoid(tf.matmul(z[t], w_z_gr_dec) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gr_dec) +
                                   b_gr_dec,
                                   name='gr_dec_{}'.format(t))
            # [B, Hd]
            h_cdd_dec[t] = tf.tanh(tf.matmul(z[t], w_z_hcdd_dec) +
                                   tf.matmul(h_dec[t - 1], w_hdec_hcdd_dec) +
                                   b_hcdd_dec,
                                   name='h_cdd_dec_{}'.format(t))
            # [B, Hd]
            h_dec[t] = tf.add(h_dec[t - 1] * (1 - gi_dec[t]),
                              h_cdd_dec[t] * gi_dec[t],
                              name='h_dec_{}'.format(t))

        #########################################################
        # Write to canvas
        #########################################################
        # [B, F, F]
        writeout[t] = tf.reshape(tf.matmul(h_dec[t], w_hdec_writeout) +
                                 b_hdec_writeout,
                                 [-1, filter_size, filter_size],
                                 name='writeout_{}'.format(t))
        
        with tf.device(device):
            # [B, H, F] * [B, F, F] * [B, F, W] = [B, H, W]
            canvas_delta[t] = tf.mul(1 / tf.exp(lg_gamma[t]), tf.batch_matmul(
                tf.batch_matmul(filter_y[t], writeout[t]), filter_x_t[t]),
                name='canvas_delta_{}'.format(t))
            # [B, H, W]
            canvas[t] = canvas[t - 1] + canvas_delta[t]

        pass

    #########################################################
    #########################################################
    # Loss
    #########################################################
    #########################################################
    x_rec = tf.sigmoid(canvas[timespan - 1], name='x_rec')
    eps = 1e-7
    ce_sum = -tf.reduce_sum(x * tf.log(x_rec + eps) +
                            (1 - x) * tf.log(1 - x_rec + eps),
                            name='ce_sum')
    ce = tf.div(ce_sum, tf.to_float(num_ex[0]), name='ce')
    lr = 1e-4
    train_step = GradientClipOptimizer(
        tf.train.AdamOptimizer(lr, epsilon=eps), clip=1.0).minimize(ce)

    m = {
        'x': x,
        'u': u,
        'x_rec': x_rec,
        'ce': ce,
        'train_step': train_step
    }

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
        'inp_height': 28,
        'inp_width': 28,
        'timespan': 20,
        'filter_size': 5,
        'num_hid_enc': 100,
        'num_hid': 20,
        'num_hid_dec': 100,
        'weight_decay': 5e-5,
        'output_dist': 'Bernoulli'
    }

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)
    m = get_train_model(opt, device=device)
    sess = tf.Session()
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

    random = np.random.RandomState(2)

    step = 0
    while step < loop_config['num_steps']:
        # Validation
        valid_ce = 0
        log.info('Running validation')
        for ii in xrange(100):
            batch = dataset.test.next_batch(100)
            x = batch[0]

            if opt['output_dist'] == 'Bernoulli':
                x = (batch[0] > 0.5).astype('float32').reshape([-1, 28, 28])
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['num_hid']])
            ce = sess.run(m['ce'], feed_dict={
                m['x']: x,
                m['u']: u
            })
            valid_ce += ce * 100 / 10000.0
        log.info('step {:d}, valid ce: {:.4f}'.format(step, valid_ce))
        valid_ce_logger.add(step, valid_ce)

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x = batch[0]

            if opt['output_dist'] == 'Bernoulli':
                x = (batch[0] > 0.5).astype('float32').reshape([-1, 28, 28])

            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['num_hid']])
            if step % 10 == 0:
                ce = sess.run(m['ce'], feed_dict={
                    m['x']: x,
                    m['u']: u
                })
                log.info('step {:d}, train ce: {:.4f}'.format(step, ce))
                train_ce_logger.add(step, ce)

            sess.run(m['train_step'], feed_dict={
                m['x']: x,
                m['u']: u
            })

            step += 1

            # Save model
            if step % 1000 == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)

    pass
