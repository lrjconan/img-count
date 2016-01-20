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
# sys.path.insert(0, '/pkgs/tensorflow-cpu-0.5.0')

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


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_autoencoder(opt, sess, train_model, device='/cpu:0'):
    """Get inference model for autoencoder."""
    #########################################################
    #########################################################
    # Constants
    #########################################################
    #########################################################
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    filter_size_r = opt['filter_size_r']
    filter_size_w = opt['filter_size_w']
    num_hid_enc = opt['num_hid_enc']
    num_hid_dec = opt['num_hid_dec']
    num_hid = opt['num_hid']

    span_x = np.reshape(np.arange(inp_width), [1, -1, 1])
    span_y = np.reshape(np.arange(inp_height), [1, -1, 1])

    span_filter_r = np.reshape(np.arange(filter_size_r), [1, 1, -1])
    span_filter_w = np.reshape(np.arange(filter_size_w), [1, 1, -1])

    #########################################################
    #########################################################
    # Variables
    #########################################################
    #########################################################
    with tf.device('/cpu:0'):
        x = tf.placeholder('float', [None, inp_width, inp_height])
        x_flat = tf.reshape(x, [-1, inp_height * inp_width], name='x_flat')

        #########################################################
        # Read attention controller
        #########################################################
        ctl_r = [None] * timespan
        ctr_x_r = [None] * timespan
        ctr_y_r = [None] * timespan
        delta_r = [None] * timespan
        lg_var_r = [None] * timespan
        lg_gamma_r = [None] * timespan

        #########################################################
        # Read Gaussian filter
        #########################################################
        mu_x_r = [None] * timespan
        mu_y_r = [None] * timespan
        filter_x_r = [None] * timespan
        filter_y_r = [None] * timespan

        #########################################################
        # Attention selector
        #########################################################
        readout_x = [None] * timespan
        readout_err = [None] * timespan
        x_and_err = [None] * timespan
        readout = [None] * timespan

        #########################################################
        # Write attention controller
        #########################################################
        ctl_w = [None] * timespan
        ctr_x_w = [None] * timespan
        ctr_y_w = [None] * timespan
        delta_w = [None] * timespan
        lg_var_w = [None] * timespan
        lg_gamma_w = [None] * timespan

        #########################################################
        # Write Gaussian filter
        #########################################################
        mu_x_w = [None] * timespan
        mu_y_w = [None] * timespan
        filter_x_w = [None] * timespan
        filter_y_w = [None] * timespan

        #########################################################
        # Write to canvas
        #########################################################
        writeout = [None] * timespan
        canvas_delta = [None] * timespan
        canvas = [None] * (timespan + 1)
        x_rec = [None] * timespan

    with tf.device(device):
        #########################################################
        # Recurrent loop
        #########################################################
        x_err = [None] * timespan
        x_err_flat = [None] * timespan

        #########################################################
        # Encoder RNN
        #########################################################
        h_enc = [None] * (timespan + 1)
        gi_enc = [None] * timespan
        gr_enc = [None] * timespan
        h_cdd_enc = [None] * timespan

        #########################################################
        # Latent distribution
        #########################################################
        z = [None] * timespan

        #########################################################
        # Decoder RNN
        #########################################################
        h_dec = [None] * (timespan + 1)
        gi_dec = [None] * timespan
        gr_dec = [None] * timespan
        h_cdd_dec = [None] * timespan

    #########################################################
    #########################################################
    # Weights
    #########################################################
    #########################################################
    with tf.device('/cpu:0'):
        #########################################################
        # Read attention Controller
        #########################################################
        w_x_ctl_r = tf.constant(sess.run(train_model['w_x_ctl_r']),
                                name='w_x_ctl_r')
        w_err_ctl_r = tf.constant(sess.run(train_model['w_err_ctl_r']),
                                  name='w_err_ctl_r')
        w_hdec_ctl_r = tf.constant(sess.run(train_model['w_hdec_ctl_r']),
                                   name='w_hdec_ctl_r')
        b_ctl_r = tf.constant(sess.run(train_model['b_ctl_r']), name='b_ctl_r')

        #########################################################
        # Write attention Controller
        #########################################################
        w_x_ctl_w = tf.constant(sess.run(train_model['w_x_ctl_w']),
                                name='w_x_ctl_w')
        w_err_ctl_w = tf.constant(sess.run(train_model['w_err_ctl_w']),
                                  name='w_err_ctl_w')
        w_hdec_ctl_w = tf.constant(sess.run(train_model['w_hdec_ctl_w']),
                                   name='w_hdec_ctl_w')
        b_ctl_w = tf.constant(sess.run(train_model['b_ctl_w']), name='b_ctl_w')

        #########################################################
        # Write to canvas
        #########################################################
        w_hdec_writeout = tf.constant(sess.run(train_model['w_hdec_writeout']),
                                      name='w_hdec_writeout')
        b_writeout = tf.constant(sess.run(train_model['b_writeout']),
                                 name='b_writeout')

    with tf.device(device):
        #########################################################
        # Encoder RNN
        #########################################################
        w_readout_gi_enc = tf.constant(
            sess.run(train_model['w_readout_gi_enc']),
            name='w_readout_gi_enc')
        w_henc_gi_enc = tf.constant(sess.run(train_model['w_henc_gi_enc']),
                                    name='w_henc_gi_enc')
        w_hdec_gi_enc = tf.constant(sess.run(train_model['w_hdec_gi_enc']),
                                    name='w_hdec_gi_enc')
        b_gi_enc = tf.constant(sess.run(train_model['b_gi_enc']),
                               name='b_gi_enc')

        w_readout_gr_enc = tf.constant(
            sess.run(train_model['w_readout_gr_enc']),
            name='w_readout_gr_enc')
        w_henc_gr_enc = tf.constant(sess.run(train_model['w_henc_gr_enc']),
                                    name='w_henc_gr_enc')
        w_hdec_gr_enc = tf.constant(sess.run(train_model['w_hdec_gr_enc']),
                                    name='w_hdec_gr_enc')
        b_gr_enc = tf.constant(sess.run(train_model['b_gr_enc']),
                               name='b_gr_enc')

        w_readout_hcdd_enc = tf.constant(
            sess.run(train_model['w_readout_hcdd_enc']),
            name='w_readout_hcdd_enc')
        w_henc_hcdd_enc = tf.constant(sess.run(train_model['w_henc_hcdd_enc']),
                                      name='w_henc_hcdd_enc')
        w_hdec_hcdd_enc = tf.constant(sess.run(train_model['w_hdec_hcdd_enc']),
                                      name='w_hdec_gr_enc')
        b_hcdd_enc = tf.constant(sess.run(train_model['b_hcdd_enc']),
                                 name='b_hcdd_enc')

        #########################################################
        # Latent distribution
        #########################################################
        w_henc_muz = tf.constant(sess.run(train_model['w_henc_muz']),
                                 name='w_henc_muz')
        b_muz = tf.constant(sess.run(train_model['b_muz']),
                            name='b_muz')

        #########################################################
        # Decoder RNN
        #########################################################
        w_z_gi_dec = tf.constant(sess.run(train_model['w_z_gi_dec']),
                                 name='w_z_gi_dec')
        w_hdec_gi_dec = tf.constant(sess.run(train_model['w_hdec_gi_dec']),
                                    name='w_hdec_gi_dec')
        b_gi_dec = tf.constant(sess.run(train_model['b_gi_dec']),
                               name='b_gi_dec')

        w_z_gr_dec = tf.constant(sess.run(train_model['w_z_gr_dec']),
                                 name='w_z_gr_dec')
        w_hdec_gr_dec = tf.constant(sess.run(train_model['w_hdec_gr_dec']),
                                    name='w_hdec_gr_dec')
        b_gr_dec = tf.constant(sess.run(train_model['b_gr_dec']),
                               name='b_gr_dec')

        w_z_hcdd_dec = tf.constant(sess.run(train_model['w_z_hcdd_dec']),
                                   name='w_z_hcdd_dec')
        w_hdec_hcdd_dec = tf.constant(sess.run(train_model['w_hdec_hcdd_dec']),
                                      name='w_hdec_hcdd_dec')
        b_hcdd_dec = tf.constant(sess.run(train_model['b_hcdd_dec']),
                                 name='b_hcdd_dec')

        # Learned initialization biases.
        if opt['learned_bias']:
            w_henc_init = tf.constant(sess.run(train_model['w_henc_init']),
                                      name='w_henc_init')
            w_hdec_init = tf.constant(sess.run(train_model['w_hdec_init']),
                                      name='w_hdec_init')
            w_canvas_init = tf.constant(sess.run(train_model['w_canvas_init']),
                                        name='w_canvas_init')

    #########################################################
    #########################################################
    # Computation graph
    #########################################################
    #########################################################
    # Setup default hidden states.
    x_shape = tf.shape(x, name='x_shape')
    num_ex = x_shape[0: 1]
    num_ex_mul = tf.concat(0, [num_ex, tf.constant([1])])
    if opt['learned_bias']:
        h_enc[-1] = tf.tile(w_henc_init, num_ex_mul, name='h_enc_-1')
        h_dec[-1] = tf.tile(w_hdec_init, num_ex_mul, name='h_dec_-1')
        canvas[-1] = w_canvas_init
    else:
        h_enc[-1] = tf.tile(tf.zeros([1, num_hid_enc]), num_ex_mul,
                            name='h_enc_-1')
        h_dec[-1] = tf.tile(tf.zeros([1, num_hid_dec]), num_ex_mul,
                            name='h_dec_-1')
        canvas[-1] = tf.zeros([inp_height, inp_width], name='canvas_-1')

    #########################################################
    # Recurrent loop
    #########################################################
    for t in xrange(timespan):
        with tf.device('/cpu:0'):
            x_err[t] = tf.sub(x, tf.sigmoid(canvas[t - 1]),
                              name='x_err_{}'.format(t))
            x_err_flat[t] = tf.reshape(x_err[t],
                                       [-1, inp_height * inp_width],
                                       name='x_err_flat_{}'.format(t))

            #########################################################
            # Read attention controller
            #########################################################
            ctl_r[t] = tf.add(tf.matmul(x_flat, w_x_ctl_r) +
                              tf.matmul(x_err_flat[t], w_err_ctl_r) +
                              tf.matmul(h_dec[t - 1], w_hdec_ctl_r),
                              b_ctl_r, name='ctl_r_{}'.format(t))
            if opt['squash']:
                ctr_x_r[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (tf.tanh(ctl_r[t][:, 0]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_r_{}'.format(t))
                ctr_y_r[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        (tf.tanh(ctl_r[t][:, 1]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_y_r_{}'.format(t))
                delta_r[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_r - 1) *
                                         tf.sigmoid(ctl_r[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_r_{}'.format(t))
            else:
                ctr_x_r[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (ctl_r[t][:, 0] + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_r_{}'.format(t))
                ctr_y_r[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        ctl_r[t][:, 1] + 1,
                                        [-1, 1, 1],
                                        name='ctr_y_r_{}'.format(t))
                delta_r[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_r - 1) *
                                         tf.exp(ctl_r[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_r_{}'.format(t))
            lg_var_r[t] = tf.reshape(ctl_r[t][:, 2], [-1, 1, 1],
                                     name='lg_var_r_{}'.format(t))
            lg_gamma_r[t] = tf.reshape(ctl_r[t][:, 4], [-1, 1, 1],
                                       name='lg_gamma_r_{}'.format(t))

            #########################################################
            # Read Gaussian filter
            #########################################################
            mu_x_r[t] = tf.add(ctr_x_r[t], delta_r[t] * (
                span_filter_r - filter_size_r / 2.0 - 0.5),
                name='mu_x_r_{}'.format(t))
            mu_y_r[t] = tf.add(ctr_y_r[t], delta_r[t] * (
                span_filter_r - filter_size_r / 2.0 - 0.5),
                name='mu_y_r_{}'.format(t))
            filter_x_r[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_r[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_x - mu_x_r[t]) * (span_x - mu_x_r[t]) /
                       tf.exp(lg_var_r[t])),
                name='filter_x_r_{}'.format(t))
            filter_y_r[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_r[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_y - mu_y_r[t]) * (span_y - mu_y_r[t]) /
                       tf.exp(lg_var_r[t])),
                name='filter_y_r_{}'.format(t))

            #########################################################
            # Read attention selector
            #########################################################
            readout_x[t] = tf.mul(tf.exp(lg_gamma_r[t]), tf.batch_matmul(
                tf.batch_matmul(filter_y_r[t], x, adj_x=True), filter_x_r[t]),
                name='readout_x_{}'.format(t))
            readout_err[t] = tf.mul(tf.exp(lg_gamma_r[t]), tf.batch_matmul(
                tf.batch_matmul(filter_y_r[t], x_err[t], adj_x=True),
                filter_x_r[t]),
                name='readout_err_{}'.format(t))
            x_and_err[t] = [tf.reshape(readout_x[t],
                                       [-1, filter_size_r * filter_size_r]),
                            tf.reshape(readout_err[t],
                                       [-1, filter_size_r * filter_size_r])]
            readout[t] = tf.concat(1, x_and_err[t],
                                   name='readout_{}'.format(t))

        with tf.device(device):
            #########################################################
            # Encoder RNN
            #########################################################
            gi_enc[t] = tf.sigmoid(tf.matmul(readout[t], w_readout_gi_enc) +
                                   tf.matmul(h_enc[t - 1], w_henc_gi_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gi_enc) +
                                   b_gi_enc,
                                   name='gi_enc_{}'.format(t))
            gr_enc[t] = tf.sigmoid(tf.matmul(readout[t], w_readout_gr_enc) +
                                   tf.matmul(h_enc[t - 1], w_henc_gr_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gr_enc) +
                                   b_gr_enc,
                                   name='gr_enc_{}'.format(t))
            h_cdd_enc[t] = tf.tanh(tf.matmul(readout[t], w_readout_hcdd_enc) +
                                   gr_enc[t] * (
                                   tf.matmul(h_enc[t - 1], w_henc_hcdd_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_hcdd_enc)) +
                                   b_hcdd_enc,
                                   name='h_cdd_enc_{}'.format(t))
            h_enc[t] = tf.add(h_enc[t - 1] * (1 - gi_enc[t]),
                              h_cdd_enc[t] * gi_enc[t],
                              name='h_enc_{}'.format(t))

            #########################################################
            # Latent distribution
            #########################################################
            z[t] = tf.add(tf.matmul(h_enc[t], w_henc_muz), b_muz,
                          name='z_{}'.format(t))

            #########################################################
            # Decoder RNN
            #########################################################
            gi_dec[t] = tf.sigmoid(tf.matmul(z[t], w_z_gi_dec) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gi_dec) +
                                   b_gi_dec,
                                   name='gi_dec_{}'.format(t))
            gr_dec[t] = tf.sigmoid(tf.matmul(z[t], w_z_gr_dec) +
                                   tf.matmul(h_dec[t - 1], w_hdec_gr_dec) +
                                   b_gr_dec,
                                   name='gr_dec_{}'.format(t))
            h_cdd_dec[t] = tf.tanh(tf.matmul(z[t], w_z_hcdd_dec) + gr_dec[t] *
                                   tf.matmul(h_dec[t - 1], w_hdec_hcdd_dec) +
                                   b_hcdd_dec,
                                   name='h_cdd_dec_{}'.format(t))
            h_dec[t] = tf.add(h_dec[t - 1] * (1 - gi_dec[t]),
                              h_cdd_dec[t] * gi_dec[t],
                              name='h_dec_{}'.format(t))

        with tf.device('/cpu:0'):
            #########################################################
            # Write attention controller
            #########################################################
            ctl_w[t] = tf.add(tf.matmul(x_flat, w_x_ctl_w) +
                              tf.matmul(x_err_flat[t], w_err_ctl_w) +
                              tf.matmul(h_dec[t], w_hdec_ctl_w),
                              b_ctl_w, name='ctl_w_{}'.format(t))
            if opt['squash']:
                ctr_x_w[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (tf.tanh(ctl_w[t][:, 0]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_w_{}'.format(t))
                ctr_y_w[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        (tf.tanh(ctl_w[t][:, 1]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_y_w_{}'.format(t))
                delta_w[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_w - 1) *
                                         tf.sigmoid(ctl_w[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_w_{}'.format(t))
            else:
                ctr_x_w[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (ctl_w[t][:, 0] + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_w_{}'.format(t))
                ctr_y_w[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        ctl_w[t][:, 1] + 1,
                                        [-1, 1, 1],
                                        name='ctr_y_w_{}'.format(t))
                delta_w[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_w - 1) *
                                         tf.exp(ctl_w[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_w_{}'.format(t))
            lg_var_w[t] = tf.reshape(ctl_w[t][:, 2], [-1, 1, 1],
                                     name='lg_var_w_{}'.format(t))
            lg_gamma_w[t] = tf.reshape(ctl_w[t][:, 4], [-1, 1, 1],
                                       name='lg_gamma_w_{}'.format(t))

            #########################################################
            # Write Gaussian filter
            #########################################################
            mu_x_w[t] = tf.add(ctr_x_w[t], delta_w[t] * (
                span_filter_w - filter_size_w / 2.0 - 0.5),
                name='mu_x_w_{}'.format(t))
            mu_y_w[t] = tf.add(ctr_y_w[t], delta_w[t] * (
                span_filter_w - filter_size_w / 2.0 - 0.5),
                name='mu_y_w_{}'.format(t))

            filter_x_w[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_w[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_x - mu_x_w[t]) * (span_x - mu_x_w[t]) /
                       tf.exp(lg_var_w[t])),
                name='filter_x_w_{}'.format(t))
            filter_y_w[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_w[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_y - mu_y_w[t]) * (span_y - mu_y_w[t]) /
                       tf.exp(lg_var_w[t])),
                name='filter_y_w_{}'.format(t))

            #########################################################
            # Write to canvas
            #########################################################
            # [B, F, Fw]
            writeout[t] = tf.reshape(tf.matmul(h_dec[t], w_hdec_writeout) +
                                     b_writeout,
                                     [-1, filter_size_w, filter_size_w],
                                     name='writeout_{}'.format(t))
            canvas_delta[t] = tf.mul(1 / tf.exp(lg_gamma_w[t]),
                                     tf.batch_matmul(
                tf.batch_matmul(filter_y_w[t], writeout[t]),
                filter_x_w[t], adj_y=True),
                name='canvas_delta_{}'.format(t))
            canvas[t] = canvas[t - 1] + canvas_delta[t]
            x_rec[t] = tf.sigmoid(canvas[t], name='x_rec_{}'.format(t))
        #########################################################
        # End of recurrent loop
        #########################################################

    m = {
        # Input
        'x': x,

        # Output
        'x_rec': x_rec,

        # Read controller
        'ctr_x_r': ctr_x_r,
        'ctr_y_r': ctr_y_r,
        'delta_r': delta_r,
        'lg_var_r': lg_var_r,
        'lg_gamma_r': lg_gamma_r,

        'mu_x_r': mu_x_r,
        'mu_y_r': mu_y_r,
        'lg_var_r': lg_var_r,
        'filter_x_r': filter_x_r,
        'filter_y_r': filter_y_r,

        'readout_x': readout_x,

        # Write controller
        'ctr_x_w': ctr_x_w,
        'ctr_y_w': ctr_y_w,
        'delta_w': delta_w,
        'lg_var_w': lg_var_w,
        'lg_gamma_w': lg_gamma_w,

        'mu_x_w': mu_x_w,
        'mu_y_w': mu_y_w,
        'lg_var_w': lg_var_w,
        'filter_x_w': filter_x_w,
        'filter_y_w': filter_y_w,

        'canvas_delta': canvas_delta
    }

    return m


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
    # Read Gaussian filter size: Fr.
    filter_size_r = opt['filter_size_r']
    # Read Gaussian filter size: Fw.
    filter_size_w = opt['filter_size_w']

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
    # Constant for computing filter_x. Shape: [1, 1, Fr].
    span_filter_r = np.reshape(np.arange(filter_size_r), [1, 1, -1])
    # Constant for computing filter_y. Shape: [1, 1, Fw].
    span_filter_w = np.reshape(np.arange(filter_size_w), [1, 1, -1])

    #########################################################
    #########################################################
    # Variables
    #########################################################
    #########################################################
    with tf.device('/cpu:0'):
        # with tf.device(device):
        # Input image. Shape: [B, T, H, W].
        x = tf.placeholder('float', [None, inp_width, inp_height])
        # [B, H * W]
        x_flat = tf.reshape(x, [-1, inp_height * inp_width], name='x_flat')

        #########################################################
        # Read attention controller
        #########################################################
        # Control variale. [g_x, g_y, log_var, log_delta, log_gamma].
        # Shape: T * [B, 5].
        ctl_r = [None] * timespan
        # Attention center x. Shape: T * [B].
        ctr_x_r = [None] * timespan
        # Attention center y. Shape: T * [B].
        ctr_y_r = [None] * timespan
        # Attention width. Shape: T * [B].
        delta_r = [None] * timespan
        # Log of the variance of Gaussian filter. Shape: T * [B].
        lg_var_r = [None] * timespan
        # Log of the multiplier of Gaussian filter. Shape: T * [B].
        lg_gamma_r = [None] * timespan

        #########################################################
        # Read Gaussian filter
        #########################################################
        # Gaussian filter mu_x. Shape: T * [B, W, Fr].
        mu_x_r = [None] * timespan
        # Gaussian filter mu_y. Shape: T * [B, H, Fr].
        mu_y_r = [None] * timespan
        # Gaussian filter on x direction. Shape: T * [B, W, Fr].
        filter_x_r = [None] * timespan
        # Gaussian filter on y direction. Shape: T * [B, H, Fr].
        filter_y_r = [None] * timespan

        #########################################################
        # Read attention selector
        #########################################################
        # Read out image from the read head. Shape: T * [B, Fr, Fr].
        readout_x = [None] * timespan
        readout_err = [None] * timespan
        x_and_err = [None] * timespan
        # Read out image from the read head. Shape: T * [B, 2 * Fr * Fr].
        readout = [None] * timespan

        #########################################################
        # Write attention controller
        #########################################################
        # Control variale. [g_x, g_y, log_var, log_delta, log_gamma].
        # Shape: T * [B, 5].
        ctl_w = [None] * timespan
        # Attention center x. Shape: T * [B].
        ctr_x_w = [None] * timespan
        # Attention center y. Shape: T * [B].
        ctr_y_w = [None] * timespan
        # Attention width. Shape: T * [B].
        delta_w = [None] * timespan
        # Log of the variance of Gaussian filter. Shape: T * [B].
        lg_var_w = [None] * timespan
        # Log of the multiplier of Gaussian filter. Shape: T * [B].
        lg_gamma_w = [None] * timespan

        #########################################################
        # Write Gaussian filter
        #########################################################
        # Gaussian filter mu_x. Shape: T * [B, W, Fw].
        mu_x_w = [None] * timespan
        # Gaussian filter mu_y. Shape: T * [B, H, Fw].
        mu_y_w = [None] * timespan
        # Gaussian filter on x direction. Shape: T * [B, W, Fw].
        filter_x_w = [None] * timespan
        # Gaussian filter on y direction. Shape: T * [B, H, Fw].
        filter_y_w = [None] * timespan

        #########################################################
        # Write to canvas
        #########################################################
        # Write out. Shape: T * [B, F, F].
        writeout = [None] * timespan
        # Add to canvas. Shape: T * [B, H, W].
        canvas_delta = [None] * timespan
        # Canvas accumulating output. Shape: T * [B, H, W].
        canvas = [None] * (timespan + 1)

    with tf.device(device):
        #########################################################
        # Recurrent loop
        #########################################################
        # Error images (original substracted by drawn). Shape: T * [B, H, W].
        x_err = [None] * timespan
        x_err_flat = [None] * timespan

        #########################################################
        # Encoder RNN
        #########################################################
        # Hidden state of the RNN encoder. Shape: (T + 1) * [B, He].
        h_enc = [None] * (timespan + 1)
        # Input gate of the RNN encoder. Shape: T * [B, He].
        gi_enc = [None] * timespan
        # Recurrent gate of the RNN encoder. Shape: T * [B, He].
        gr_enc = [None] * timespan
        # Hidden candidate activation of the RNN encoder.
        h_cdd_enc = [None] * timespan

        #########################################################
        # Latent distribution
        #########################################################
        # Noise. Shape: [B, T, Hz]
        u = tf.placeholder('float', [None, timespan, num_hid])
        u_l = tf.split(1, timespan, u)
        u_l_flat = [None] * timespan

        # Mean of hidden latent variable. Shape: T * [B, Hz].
        mu_z = [None] * timespan
        # Standard deviation of hidden latent variable. Shape: T * [B, Hz].
        lg_sigma_z = [None] * timespan
        sigma_z = [None] * timespan
        # Hidden latent variable. Shape: T * [B, Hz].
        z = [None] * timespan
        # KL divergence
        kl_qzx_pz = [None] * timespan

        #########################################################
        # Decoder RNN
        #########################################################
        # Hidden state of the RNN decoder. Shape: (T + 1) * [B, Hd].
        h_dec = [None] * (timespan + 1)
        # Input gate of the RNN decoder. Shape: T * [B, Hd].
        gi_dec = [None] * timespan
        # Recurrent gate of the RNN decoder. Shape: T * [B, Hd].
        gr_dec = [None] * timespan
        # Hidden candidate activation of the RNN decoder. Shape: T * [B, Hd].
        h_cdd_dec = [None] * timespan

    #########################################################
    #########################################################
    # Weights
    #########################################################
    #########################################################
    with tf.device('/cpu:0'):
        #########################################################
        # Read attention controller
        #########################################################
        # From input image to controller variables. Shape: [H * W, 5].
        w_x_ctl_r = weight_variable([inp_height * inp_width, 5], wd=wd,
                                    name='w_x_ctl_r')
        # From err image to controller variables. Shape: [H * W, 5].
        w_err_ctl_r = weight_variable([inp_height * inp_width, 5], wd=wd,
                                      name='w_err_ctl_r')
        # From hidden decoder to controller variables. Shape: [Hd, 5].
        w_hdec_ctl_r = weight_variable(
            [num_hid_dec, 5], wd=wd, name='w_hdec_ctl_r')
        # Controller variable bias. Shape: [5].
        b_ctl_r = weight_variable([5], wd=wd, name='b_ctl_r')

        #########################################################
        # Write attention controller
        #########################################################
        # From input image to controller variables. Shape: [H * W, 5].
        w_x_ctl_w = weight_variable([inp_height * inp_width, 5], wd=wd,
                                    name='w_x_ctl_w')
        # From err image to controller variables. Shape: [H * W, 5].
        w_err_ctl_w = weight_variable([inp_height * inp_width, 5], wd=wd,
                                      name='w_err_ctl_w')
        # From hidden decoder to controller variables. Shape: [Hd, 5].
        w_hdec_ctl_w = weight_variable([num_hid_dec, 5], wd=wd,
                                       name='w_hdec_ctl_w')
        # Controller variable bias. Shape: [5].
        b_ctl_w = weight_variable([5], wd=wd, name='b_ctl_w')

        #########################################################
        # Write to canvas
        #########################################################
        # From decoder to write. Shape: [Hd, Fw * Fw].
        w_hdec_writeout = weight_variable(
            [num_hid_dec, filter_size_w * filter_size_w], wd=wd,
            name='w_hdec_writeout')
        # Shape: [Fw * Fw].
        b_writeout = weight_variable(
            [filter_size_w * filter_size_w], wd=wd, name='b_writeout')

    with tf.device(device):
        #########################################################
        # Encoder RNN
        #########################################################
        # From read out to input gate encoder. Shape: [2 * F * F, He].
        w_readout_gi_enc = weight_variable(
            [2 * filter_size_r * filter_size_r, num_hid_enc], wd=wd,
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
            [2 * filter_size_r * filter_size_r, num_hid_enc], wd=wd,
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
            [2 * filter_size_r * filter_size_r, num_hid_enc], wd=wd,
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
        w_henc_lgsigmaz = weight_variable(
            [num_hid_enc, num_hid], wd=wd, name='w_henc_lgsigmaz')
        b_lgsigmaz = weight_variable([num_hid], wd=wd, name='b_lgsigmaz')

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

        # Learned initialization biases.
        if opt['learned_bias']:
            w_henc_init = weight_variable([1, num_hid_enc], wd=wd,
                                          name='w_henc_init')
            w_hdec_init = weight_variable([1, num_hid_dec], wd=wd,
                                          name='w_hdec_init')
            w_canvas_init = weight_variable([inp_height, inp_width], wd=wd,
                                            name='w_canvas_init')
    #########################################################
    #########################################################
    # Computation graph
    #########################################################
    #########################################################
    # Setup default hidden states.
    x_shape = tf.shape(x, name='x_shape')
    num_ex = x_shape[0: 1]
    num_ex_mul = tf.concat(0, [num_ex, tf.constant([1])])

    if opt['learned_bias']:
        h_enc[-1] = tf.tile(w_henc_init, num_ex_mul, name='h_enc_-1')
        h_dec[-1] = tf.tile(w_hdec_init, num_ex_mul, name='h_dec_-1')
        canvas[-1] = w_canvas_init
    else:
        h_enc[-1] = tf.tile(tf.zeros([1, num_hid_enc]), num_ex_mul,
                            name='h_enc_-1')
        h_dec[-1] = tf.tile(tf.zeros([1, num_hid_dec]), num_ex_mul,
                            name='h_dec_-1')
        canvas[-1] = tf.zeros([inp_height, inp_width], name='canvas_-1')

    #########################################################
    # Recurrent loop
    #########################################################
    for t in xrange(timespan):
        with tf.device('/cpu:0'):
            # [B, H * W]
            x_err[t] = tf.sub(x, tf.sigmoid(canvas[t - 1]),
                              name='x_err_{}'.format(t))
            x_err_flat[t] = tf.reshape(x_err[t],
                                       [-1, inp_height * inp_width],
                                       name='x_err_flat_{}'.format(t))
            u_l_flat[t] = tf.reshape(u_l[t], [-1, num_hid],
                                     name='u_flat_{}'.format(t))

            #########################################################
            # Read attention controller
            #########################################################
            # [B, 5]
            ctl_r[t] = tf.add(tf.matmul(x_flat, w_x_ctl_r) +
                              tf.matmul(x_err_flat[t], w_err_ctl_r) +
                              tf.matmul(h_dec[t - 1], w_hdec_ctl_r),
                              b_ctl_r, name='ctl_r_{}'.format(t))
            if opt['squash']:
                # Squash in (-1, 1), [B, 1, 1]
                ctr_x_r[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (tf.tanh(ctl_r[t][:, 0]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_r_{}'.format(t))
                ctr_y_r[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        (tf.tanh(ctl_r[t][:, 1]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_y_r_{}'.format(t))
                # Squash in (0, 1), [B, 1, 1]
                delta_r[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_r - 1) *
                                         tf.sigmoid(ctl_r[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_r_{}'.format(t))
            else:
                # [B, 1, 1]
                ctr_x_r[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (ctl_r[t][:, 0] + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_r_{}'.format(t))
                # [B, 1, 1]
                ctr_y_r[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        (ctl_r[t][:, 1] + 1),
                                        [-1, 1, 1],
                                        name='ctr_y_r_{}'.format(t))

                # [B, 1, 1]
                delta_r[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_r - 1) *
                                         tf.exp(ctl_r[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_r_{}'.format(t))
            # [B, 1, 1]
            lg_var_r[t] = tf.reshape(ctl_r[t][:, 2], [-1, 1, 1],
                                     name='lg_var_r_{}'.format(t))
            # [B, 1, 1]
            lg_gamma_r[t] = tf.reshape(ctl_r[t][:, 4], [-1, 1, 1],
                                       name='lg_gamma_r_{}'.format(t))

            #########################################################
            # Read Gaussian filter
            #########################################################
            # [B, 1, 1] + [B, 1, 1] * [1, Fr, 1] = [B, 1, Fr]
            mu_x_r[t] = tf.add(ctr_x_r[t], delta_r[t] * (
                span_filter_r - filter_size_r / 2.0 - 0.5),
                name='mu_x_r_{}'.format(t))
            # [B, 1, 1] + [B, 1, 1] * [1, 1, Fr] = [B, 1, Fr]
            mu_y_r[t] = tf.add(ctr_y_r[t], delta_r[t] * (
                span_filter_r - filter_size_r / 2.0 - 0.5),
                name='mu_y_r_{}'.format(t))

            # [B, 1, 1] * [1, W, 1] - [B, 1, Fr] = [B, W, Fr]
            filter_x_r[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_r[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_x - mu_x_r[t]) * (span_x - mu_x_r[t]) /
                       tf.exp(lg_var_r[t])),
                name='filter_x_r_{}'.format(t))
            # [1, H, 1] - [B, 1, Fr] = [B, H, Fr]
            filter_y_r[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_r[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_y - mu_y_r[t]) * (span_y - mu_y_r[t]) /
                       tf.exp(lg_var_r[t])),
                name='filter_y_r_{}'.format(t))

            #########################################################
            # Read attention selector
            #########################################################
            # [B, 1, 1] * [B, F, H] * [B, H, W] * [B, W, F] = [B, F, F]
            readout_x[t] = tf.mul(tf.exp(lg_gamma_r[t]), tf.batch_matmul(
                tf.batch_matmul(filter_y_r[t], x, adj_x=True), filter_x_r[t]),
                name='readout_x_{}'.format(t))
            readout_err[t] = tf.mul(tf.exp(lg_gamma_r[t]), tf.batch_matmul(
                tf.batch_matmul(filter_y_r[t], x_err[t], adj_x=True),
                filter_x_r[t]),
                name='readout_err_{}'.format(t))

            # [B, 2 * F]
            x_and_err[t] = [tf.reshape(readout_x[t],
                                       [-1, filter_size_r * filter_size_r]),
                            tf.reshape(readout_err[t],
                                       [-1, filter_size_r * filter_size_r])]
            readout[t] = tf.concat(1, x_and_err[t],
                                   name='readout_{}'.format(t))

        with tf.device(device):
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
                                   gr_enc[t] * (
                                   tf.matmul(h_enc[t - 1], w_henc_hcdd_enc) +
                                   tf.matmul(h_dec[t - 1], w_hdec_hcdd_enc)) +
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
            lg_sigma_z[t] = tf.add(tf.matmul(h_enc[t], w_henc_lgsigmaz),
                                   b_lgsigmaz,
                                   name='lg_sigma_z_{}'.format(t))
            sigma_z[t] = tf.exp(lg_sigma_z[t], name='sigma_z_{}'.format(t))
            z[t] = tf.add(mu_z[t], sigma_z[t] * u_l_flat[t],
                          name='z_{}'.format(t))
            # KL Divergence
            kl_qzx_pz[t] = tf.mul(-0.5,
                                  tf.reduce_sum(1 + 2 * lg_sigma_z[t] -
                                                mu_z[t] * mu_z[t] -
                                                tf.exp(2 * lg_sigma_z[t])),
                                  name='kl_qzx_pz')

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
            h_cdd_dec[t] = tf.tanh(tf.matmul(z[t], w_z_hcdd_dec) + gr_dec[t] *
                                   tf.matmul(h_dec[t - 1], w_hdec_hcdd_dec) +
                                   b_hcdd_dec,
                                   name='h_cdd_dec_{}'.format(t))
            # [B, Hd]
            h_dec[t] = tf.add(h_dec[t - 1] * (1 - gi_dec[t]),
                              h_cdd_dec[t] * gi_dec[t],
                              name='h_dec_{}'.format(t))

        with tf.device('/cpu:0'):
            #########################################################
            # Write attention controller
            #########################################################
            # [B, 5]
            ctl_w[t] = tf.add(tf.matmul(x_flat, w_x_ctl_w) +
                              tf.matmul(x_err_flat[t], w_err_ctl_w) +
                              tf.matmul(h_dec[t], w_hdec_ctl_w),
                              b_ctl_w, name='ctl_w_{}'.format(t))
            if opt['squash']:
                # [B, 1, 1]
                ctr_x_w[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (tf.tanh(ctl_w[t][:, 0]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_w_{}'.format(t))
                # [B, 1, 1]
                ctr_y_w[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        (tf.tanh(ctl_w[t][:, 1]) + 1),
                                        [-1, 1, 1],
                                        name='ctr_y_w_{}'.format(t))
                # [B, 1, 1]
                delta_w[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_w - 1) *
                                         tf.sigmoid(ctl_w[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_w_{}'.format(t))
            else:
                # [B, 1, 1]
                ctr_x_w[t] = tf.reshape((inp_width + 1) / 2.0 *
                                        (ctl_w[t][:, 0] + 1),
                                        [-1, 1, 1],
                                        name='ctr_x_w_{}'.format(t))
                # [B, 1, 1]
                ctr_y_w[t] = tf.reshape((inp_height + 1) / 2.0 *
                                        (ctl_w[t][:, 1] + 1),
                                        [-1, 1, 1],
                                        name='ctr_y_w_{}'.format(t))
                # [B, 1, 1]
                delta_w[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                                        ((filter_size_w - 1) *
                                         tf.exp(ctl_w[t][:, 3])),
                                        [-1, 1, 1],
                                        name='delta_w_{}'.format(t))
            # [B, 1, 1]
            lg_var_w[t] = tf.reshape(ctl_w[t][:, 2], [-1, 1, 1],
                                     name='lg_var_w_{}'.format(t))
            # [B, 1, 1]
            lg_gamma_w[t] = tf.reshape(ctl_w[t][:, 4], [-1, 1, 1],
                                       name='lg_gamma_w_{}'.format(t))

            #########################################################
            # Write Gaussian filter
            #########################################################
            # [B, 1, 1] + [B, 1, 1] * [1, Fw, 1] = [B, 1, Fw]
            mu_x_w[t] = tf.add(ctr_x_w[t], delta_w[t] * (
                span_filter_w - filter_size_w / 2.0 - 0.5),
                name='mu_x_w_{}'.format(t))
            # [B, 1, 1] + [B, 1, 1] * [1, 1, Fw] = [B, 1, Fw]
            mu_y_w[t] = tf.add(ctr_y_w[t], delta_w[t] * (
                span_filter_w - filter_size_w / 2.0 - 0.5),
                name='mu_y_w_{}'.format(t))

            # [B, 1, 1] * [1, W, 1] - [B, 1, Fw] = [B, W, Fw]
            filter_x_w[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_w[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_x - mu_x_w[t]) * (span_x - mu_x_w[t]) /
                       tf.exp(lg_var_w[t])),
                name='filter_x_w_{}'.format(t))
            # [1, H, 1] - [B, 1, Fw] = [B, H, Fw]
            filter_y_w[t] = tf.mul(
                1 / tf.sqrt(tf.exp(lg_var_w[t])) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span_y - mu_y_w[t]) * (span_y - mu_y_w[t]) /
                       tf.exp(lg_var_w[t])),
                name='filter_y_w_{}'.format(t))

            #########################################################
            # Write to canvas
            #########################################################
            # [B, F, Fw]
            writeout[t] = tf.reshape(tf.matmul(h_dec[t], w_hdec_writeout) +
                                     b_writeout,
                                     [-1, filter_size_w, filter_size_w],
                                     name='writeout_{}'.format(t))

            # [B, H, Fw] * [B, Fw, Fw] * [B, F, W] = [B, H, W]
            canvas_delta[t] = tf.mul(1 / tf.exp(lg_gamma_w[t]),
                                     tf.batch_matmul(
                tf.batch_matmul(filter_y_w[t], writeout[t]),
                filter_x_w[t], adj_y=True),
                name='canvas_delta_{}'.format(t))
            # [B, H, W]
            canvas[t] = canvas[t - 1] + canvas_delta[t]
        #########################################################
        # End of recurrent loop
        #########################################################

    #########################################################
    #########################################################
    # Loss and gradient
    #########################################################
    #########################################################
    with tf.device('/cpu:0'):
        x_rec = tf.sigmoid(canvas[timespan - 1], name='x_rec')
        eps = 1e-7
        kl_qzx_pz_sum = tf.reduce_sum(tf.concat(0, kl_qzx_pz))
        log_pxz_sum = tf.reduce_sum(x * tf.log(x_rec + eps) +
                                    (1 - x) * tf.log(1 - x_rec + eps),
                                    name='ce_sum')
        w_kl = 1.0
        w_logp = 1.0

        # Cross entropy normalized by number of examples.
        ce = tf.div(-log_pxz_sum, tf.to_float(num_ex[0]), name='ce')

        # Lower bound
        log_px_lb = tf.div(-w_kl * kl_qzx_pz_sum +
                           w_logp * log_pxz_sum /
                           (w_kl + w_logp) * 2.0, tf.to_float(num_ex[0]),
                           name='log_px_lb')
        tf.add_to_collection('losses', -log_px_lb)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        lr = 1e-4
        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(lr, epsilon=eps), clip=1.0).minimize(
            total_loss)

    m = {
        # Input
        'x': x,
        'u': u,

        # Output
        'x_rec': x_rec,

        # Loss
        'ce': ce,

        # Train
        'train_step': train_step,

        # Weights
        'w_x_ctl_r': w_x_ctl_r,
        'w_err_ctl_r': w_err_ctl_r,
        'w_hdec_ctl_r': w_hdec_ctl_r,
        'b_ctl_r': b_ctl_r,

        'w_x_ctl_w': w_x_ctl_w,
        'w_err_ctl_w': w_err_ctl_w,
        'w_hdec_ctl_w': w_hdec_ctl_w,
        'b_ctl_w': b_ctl_w,

        'w_readout_gi_enc': w_readout_gi_enc,
        'w_henc_gi_enc': w_henc_gi_enc,
        'w_hdec_gi_enc': w_hdec_gi_enc,
        'b_gi_enc': b_gi_enc,

        'w_readout_gr_enc': w_readout_gr_enc,
        'w_henc_gr_enc': w_henc_gr_enc,
        'w_hdec_gr_enc': w_hdec_gr_enc,
        'b_gr_enc': b_gr_enc,

        'w_readout_hcdd_enc': w_readout_hcdd_enc,
        'w_henc_hcdd_enc': w_henc_hcdd_enc,
        'w_hdec_hcdd_enc': w_hdec_hcdd_enc,
        'b_hcdd_enc': b_hcdd_enc,

        'w_henc_muz': w_henc_muz,
        'b_muz': b_muz,
        'w_henc_lgsigmaz': w_henc_lgsigmaz,
        'b_lgsigmaz': b_lgsigmaz,

        'w_z_gi_dec': w_z_gi_dec,
        'w_hdec_gi_dec': w_hdec_gi_dec,
        'b_gi_dec': b_gi_dec,

        'w_z_gr_dec': w_z_gr_dec,
        'w_hdec_gr_dec': w_hdec_gr_dec,
        'b_gr_dec': b_gr_dec,

        'w_z_hcdd_dec': w_z_hcdd_dec,
        'w_hdec_hcdd_dec': w_hdec_hcdd_dec,
        'b_hcdd_dec': b_hcdd_dec,

        'w_hdec_writeout': w_hdec_writeout,
        'b_writeout': b_writeout
    }
    if opt['learned_bias']:
        m['w_henc_init'] = w_henc_init
        m['w_hdec_init'] = w_hdec_init
        m['w_canvas_init'] = w_canvas_init

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
    args = parser.parse_args()

    return args


def preprocess(x, opt):
    if opt['output_dist'] == 'Bernoulli':
        return (batch[0] > 0.5).astype('float32').reshape([-1, 28, 28])
    else:
        return x.reshape([-1, 28, 28])

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
        'timespan': 64,
        'filter_size_r': 2,
        'filter_size_w': 5,
        'num_hid_enc': 256,
        'num_hid': 100,
        'num_hid_dec': 256,
        'weight_decay': 5e-5,
        'output_dist': 'Bernoulli',
        'learned_bias': True,
        'squash': True
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
            x = preprocess(batch[0])
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['num_hid']])
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
            x = preprocess(batch[0])
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['num_hid']])
            if step % 10 == 0:
                ce = sess.run(m['ce'], feed_dict={
                    m['x']: x,
                    m['u']: u
                })
                log.info('step {:d}, train ce {:.4f}'.format(step, ce))
                train_ce_logger.add(step, ce)

            sess.run(m['train_step'], feed_dict={
                m['x']: x,
                m['u']: u
            })

            step += 1

            # Save model
            if step % args.steps_per_ckpt == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)

    sess.close()
    pass
