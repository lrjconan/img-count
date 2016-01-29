"""
This code implements DRAW (Deep Recurrent Attention Writer) [2] on MNIST.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:

Reference:
[1] K. Gregor, I. Danihelka, A. Grabes, D.J. Rezende, D. Wierstra. DRAW: A
recurrent neural network for image generation. ICML 2015.
"""
import cslab_environ

from data_api import mnist
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
from utils.time_series_logger import TimeSeriesLogger
import argparse
import datetime
import fnmatch
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time

log = logger.get()


def _get_latest_ckpt(folder):
    """Get the latest checkpoint filename in a folder."""

    ckpt_fname_pattern = os.path.join(folder, 'model.ckpt-*')
    ckpt_fname_list = []
    for fname in os.listdir(folder):
        fullname = os.path.join(folder, fname)
        if fnmatch.fnmatch(fullname, ckpt_fname_pattern):
            ckpt_fname_list.append(fullname)
    if len(ckpt_fname_list) == 0:
        raise Exception('No checkpoint file found.')
    ckpt_fname_step = [int(fn.split('-')[-1]) for fn in ckpt_fname_list]
    latest_step = max(ckpt_fname_step)

    return os.path.join(folder, 'model.ckpt-{}'.format(latest_step)), latest_step


def weight_variable(shape, wd=None, name=None, sess=None, train_model=None):
    """Initialize weights."""
    if sess is None:
        initial = tf.truncated_normal(shape, stddev=0.01)
        var = tf.Variable(initial, name=name)
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
    else:
        return tf.constant(sess.run(train_model[name]))


def _add_gated_rnn(model, timespan, inp_dim, hid_dim, h_init, wd=None, name='', sess=None, train_model=None):
    h = [None] * (timespan + 1)
    g_i = [None] * timespan
    g_r = [None] * timespan
    h[-1] = h_init

    w_xi = weight_variable(
        [inp_dim, hid_dim], wd=wd, name='w_xi_{}'.format(name),
        sess=sess, train_model=train_model)
    w_hi = weight_variable(
        [hid_dim, hid_dim], wd=wd, name='w_hi_{}'.format(name),
        sess=sess, train_model=train_model)
    b_i = weight_variable(
        [hid_dim], wd=wd, name='b_i_{}'.format(name),
        sess=sess, train_model=train_model)

    w_xu = weight_variable(
        [inp_dim, hid_dim], wd=wd, name='w_xu_{}'.format(name),
        sess=sess, train_model=train_model)
    w_hu = weight_variable(
        [hid_dim, hid_dim], wd=wd, name='w_hu_{}'.format(name),
        sess=sess, train_model=train_model)
    b_u = weight_variable(
        [hid_dim], wd=wd, name='b_u_{}'.format(name),
        sess=sess, train_model=train_model)

    w_xr = weight_variable(
        [inp_dim, hid_dim], wd=wd, name='w_xr_{}'.format(name),
        sess=sess, train_model=train_model)
    w_hr = weight_variable(
        [hid_dim, hid_dim], wd=wd, name='w_hr_{}'.format(name),
        sess=sess, train_model=train_model)
    b_r = weight_variable(
        [hid_dim], wd=wd, name='b_r_{}'.format(name),
        sess=sess, train_model=train_model)

    def unroll(inp, time):
        t = time

        # [B, H]
        g_i[t] = tf.sigmoid(tf.matmul(inp, w_xi) +
                            tf.matmul(h[t - 1], w_hi) +
                            b_i,
                            name='g_i_{}_{}'.format(name, t))
        # [B, H]
        g_r[t] = tf.sigmoid(tf.matmul(inp, w_xr) +
                            tf.matmul(h[t - 1], w_hr) +
                            b_r,
                            name='g_r_{}_{}'.format(name, t))
        # [B, H]
        u = tf.tanh(tf.matmul(inp, w_xu) + g_r[t] *
                    (tf.matmul(h[t - 1], w_hu)) +
                    b_u,
                    name='u_{}_{}'.format(name, t))
        # [B, He]
        h[t] = tf.add(h[t - 1] * (1 - g_i[t]), u * g_i[t],
                      name='h_{}_{}'.format(name, t))

    model['w_xi_{}'.format(name)] = w_xi
    model['w_hi_{}'.format(name)] = w_hi
    model['b_i_{}'.format(name)] = b_i
    model['w_xu_{}'.format(name)] = w_xu
    model['w_hu_{}'.format(name)] = w_hu
    model['b_u_{}'.format(name)] = b_u
    model['w_xr_{}'.format(name)] = w_xr
    model['w_hr_{}'.format(name)] = w_hr
    model['b_r_{}'.format(name)] = b_r
    model['g_i_{}'.format(name)] = g_i
    model['g_r_{}'.format(name)] = g_i
    model['h_{}'.format(name)] = h

    return unroll


def _add_controller_rnn(model, timespan, inp_width, inp_height, ctl_inp_dim, filter_size, wd=None, name='', sess=None, train_model=None):
    """Add an attention controller."""
    # Constant for computing filter_x. Shape: [1, W, 1].
    span_x = np.reshape(np.arange(inp_width), [1, inp_width, 1])
    # Constant for computing filter_y. Shape: [1, H, 1].
    span_y = np.reshape(np.arange(inp_height), [1, inp_height, 1])
    # Constant for computing filter_x. Shape: [1, 1, Fr].
    span_filter = np.reshape(np.arange(filter_size) + 1, [1, 1, -1])

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
    # Gaussian filter mu_x. Shape: T * [B, W, Fr].
    mu_x = [None] * timespan
    # Gaussian filter mu_y. Shape: T * [B, H, Fr].
    mu_y = [None] * timespan
    # Gaussian filter on x direction. Shape: T * [B, W, Fr].
    filter_x = [None] * timespan
    # Gaussian filter on y direction. Shape: T * [B, H, Fr].
    filter_y = [None] * timespan

    # From hidden decoder to controller variables. Shape: [Hd, 5].
    w_ctl = weight_variable(
        [ctl_inp_dim, 5], wd=wd, name='w_ctl_{}'.format(name),
        sess=sess, train_model=train_model)
    # Controller variable bias. Shape: [5].
    b_ctl = weight_variable([5], wd=wd, name='b_ctl_{}'.format(name),
                            sess=sess, train_model=train_model)

    def unroll(ctl_inp, time):
        """Run controller."""
        t = time
        #########################################################
        # Attention controller
        #########################################################
        # [B, 5]
        ctl[t] = tf.matmul(ctl_inp, w_ctl) + b_ctl
        # [B, 1, 1]
        ctr_x[t] = tf.reshape((inp_width + 1) / 2.0 *
                              (ctl[t][:, 0] + 1),
                              [-1, 1, 1],
                              name='ctr_x_{}_{}'.format(name, t))
        # [B, 1, 1]
        ctr_y[t] = tf.reshape((inp_height + 1) / 2.0 *
                              (ctl[t][:, 1] + 1),
                              [-1, 1, 1],
                              name='ctr_y_{}_{}'.format(name, t))
        # [B, 1, 1]
        delta[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                              ((filter_size - 1) *
                               tf.exp(ctl[t][:, 3])),
                              [-1, 1, 1],
                              name='delta_{}_{}'.format(name, t))
        # [B, 1, 1]
        lg_var[t] = tf.reshape(ctl[t][:, 2], [-1, 1, 1],
                               name='lg_var_{}_{}'.format(name, t))
        # [B, 1, 1]
        lg_gamma[t] = tf.reshape(ctl[t][:, 4], [-1, 1, 1],
                                 name='lg_gamma_{}_{}'.format(name, t))

        #########################################################
        # Gaussian filter
        #########################################################
        # [B, 1, 1] + [B, 1, 1] * [1, Fr, 1] = [B, 1, Fr]
        mu_x[t] = tf.add(ctr_x[t], delta[t] * (
            span_filter - filter_size / 2.0 - 0.5),
            name='mu_x_{}_{}'.format(name, t))
        # [B, 1, 1] + [B, 1, 1] * [1, 1, Fr] = [B, 1, Fr]
        mu_y[t] = tf.add(ctr_y[t], delta[t] * (
            span_filter - filter_size / 2.0 - 0.5),
            name='mu_y_{}_{}'.format(name, t))
        # [B, 1, 1] * [1, W, 1] - [B, 1, Fr] = [B, W, Fr]
        filter_x[t] = tf.mul(
            1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi),
            tf.exp(-0.5 * (span_x - mu_x[t]) * (span_x - mu_x[t]) /
                   tf.exp(lg_var[t])),
            name='filter_x_{}_{}'.format(name, t))
        # [1, H, 1] - [B, 1, Fr] = [B, H, Fr]
        filter_y[t] = tf.mul(
            1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi),
            tf.exp(-0.5 * (span_y - mu_y[t]) * (span_y - mu_y[t]) /
                   tf.exp(lg_var[t])),
            name='filter_y_{}_{}'.format(name, t))

        pass

    model['w_ctl_{}'.format(name)] = w_ctl
    model['b_ctl_{}'.format(name)] = b_ctl
    model['ctl_{}'.format(name)] = ctl
    model['ctr_x_{}'.format(name)] = ctr_x
    model['ctr_y_{}'.format(name)] = ctr_y
    model['delta_{}'.format(name)] = delta
    model['lg_var_{}'.format(name)] = lg_var
    model['lg_gamma_{}'.format(name)] = lg_gamma
    model['mu_x_{}'.format(name)] = mu_x
    model['mu_y_{}'.format(name)] = mu_y
    model['filter_x_{}'.format(name)] = filter_x
    model['filter_y_{}'.format(name)] = filter_y

    return unroll


def _batch_matmul(x, y, rep_x=True, adj_x=False, adj_y=False):
    """Same as tf.batch_matmul, but GPU friendly.

    Args:
        x: first tensor, [B, R1, C1]
        y: second tensor, [B, R2, C2]
        rep_x: tensorflow is not fully broadcasting so you need to decide
        whether to repeat x or repeat y, repeat x by default.
        adj_x: whether to transpose x.
        adj_y: whether to transpose y.

    Returns:
        z: resulting tensor, [B, R1, C2] (without transposing).
    """
    if adj_x and adj_y:
        # [B, H, F] * [B, W, H] = [B, F, W]
        raise Exception('Not supported.')
    elif adj_x:
        # [B, H, F] * [B, H, W] = [B, F, W]
        x2 = tf.expand_dims(x, dim=3)
        y2 = tf.expand_dims(y, dim=2)
        if rep_x:
            t = tf.constant([1, 1, 1, y.get_shape()[2].value])
            x3 = tf.tile(x2, t)
            y3 = y2
        else:
            t = tf.constant([1, 1, x.get_shape()[2].value, 1])
            x3 = x2
            y3 = tf.tile(y2, t)
        return tf.reduce_sum(x3 * y3, reduction_indices=[1])
    elif adj_y:
        # [B, F, H] * [B, W, H] = [B, F, W]
        x2 = tf.expand_dims(x, dim=2)
        y2 = tf.expand_dims(y, dim=1)
        if rep_x:
            t = tf.constant([1, 1, y.get_shape()[1].value, 1])
            x3 = tf.tile(x2, t)
            y3 = y2
        else:
            t = tf.constant([1, x.get_shape()[1].value, 1, 1])
            x3 = x2
            y3 = tf.tile(y2, t)
        return tf.reduce_sum(x3 * y3, reduction_indices=[3])
    else:
        # [B, F, H] * [B, H, W] = [B, F, W]
        x2 = tf.expand_dims(x, dim=3)
        y2 = tf.expand_dims(y, dim=1)
        if rep_x:
            t = tf.constant([1, 1, 1, y.get_shape()[2].value])
            x3 = tf.tile(x2, t)
            y3 = y2
        else:
            t = tf.constant([1, x.get_shape()[1].value, 1, 1])
            x3 = x2
            y3 = tf.tile(y2, t)
        return tf.reduce_sum(x3 * y3, reduction_indices=[2])


def get_generator(opt, sess, train_model, device='/cpu:0'):
    """Get generator model."""
    m = {}
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    filter_size_w = opt['filter_size_w']
    hid_dec_dim = opt['hid_dec_dim']
    hid_dim = opt['hid_dim']
    wd = opt['weight_decay']

    with tf.device(device):
        #########################################################
        # Variables
        #########################################################
        # Write attention controller
        unroll_write_controller = _add_controller_rnn(
            model=m,
            timespan=timespan,
            inp_width=inp_width,
            inp_height=inp_height,
            ctl_inp_dim=hid_dec_dim,
            filter_size=filter_size_w,
            wd=wd,
            name='w',
            sess=sess,
            train_model=train_model
        )
        lg_gamma_w = m['lg_gamma_w']
        filter_x_w = m['filter_x_w']
        filter_y_w = m['filter_y_w']

        # Write to canvas
        writeout = [None] * timespan
        w_hdec_writeout = weight_variable(
            [hid_dec_dim, filter_size_w * filter_size_w], wd=wd,
            name='w_hdec_writeout',
            sess=sess, train_model=train_model)
        b_writeout = weight_variable(
            [filter_size_w * filter_size_w], wd=wd, name='b_writeout',
            sess=sess, train_model=train_model)
        m['w_hdec_writeout'] = w_hdec_writeout
        m['b_writeout'] = b_writeout

        canvas_delta = [None] * timespan
        m['canvas_delta'] = canvas_delta

        canvas = [None] * (timespan + 1)
        w_canvas_init = weight_variable([inp_height, inp_width], wd=wd,
                                        name='w_canvas_init',
                                        sess=sess, train_model=train_model)
        canvas[-1] = w_canvas_init
        m['w_canvas_init'] = w_canvas_init

        x_rec = [None] * timespan
        m['x_rec'] = x_rec

        z = tf.placeholder('float', [None, timespan, hid_dim])
        m['z'] = z
        z_l = tf.split(1, timespan, z)
        z_shape = tf.shape(z, name='z_shape')
        num_ex = z_shape[0: 1]
        num_ex_mul = tf.concat(0, [num_ex, tf.constant([1])])

        w_hdec_init = weight_variable([1, hid_dec_dim], wd=wd,
                                      name='w_hdec_init',
                                      sess=sess, train_model=train_model)
        hdec_init = tf.tile(w_hdec_init, num_ex_mul, name='hdec_init')
        m['w_hdec_init'] = w_hdec_init
        unroll_decoder = _add_gated_rnn(
            model=m,
            timespan=timespan,
            inp_dim=hid_dim,
            hid_dim=hid_dec_dim,
            h_init=hdec_init,
            name='dec',
            sess=sess,
            train_model=train_model
        )
        h_dec = m['h_dec']

        #########################################################
        # Computation graph
        #########################################################
        for t in xrange(timespan):
            unroll_decoder(inp=tf.reshape(z_l[t], [-1, hid_dim]), time=t)
            unroll_write_controller(ctl_inp=h_dec[t], time=t)

            writeout[t] = tf.reshape(tf.matmul(h_dec[t], w_hdec_writeout) +
                                     b_writeout,
                                     [-1, filter_size_w, filter_size_w],
                                     name='writeout_{}'.format(t))

            canvas_delta[t] = tf.mul(1 / tf.exp(lg_gamma_w[t]),
                                     _batch_matmul(_batch_matmul(
                                         filter_y_w[t], writeout[t],
                                         rep_x=False),
                                     filter_x_w[t], adj_y=True),
                                     name='canvas_delta_{}'.format(t))
            canvas[t] = canvas[t - 1] + canvas_delta[t]
            x_rec[t] = tf.sigmoid(canvas[t], name='x_rec')

    return m


def get_model(opt, device='/cpu:0', train=True):
    """Get train model for DRAW."""
    m = {}
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
    hid_enc_dim = opt['hid_enc_dim']
    # Number of hidden units in the RNN decoder: Hd.
    hid_dec_dim = opt['hid_dec_dim']
    # Number of hidden dimension in the latent variable: Hz.
    hid_dim = opt['hid_dim']

    # Weight L2 regularization.
    wd = opt['weight_decay']

    with tf.device(device):
        #########################################################
        # Variables
        #########################################################
        # Input image. Shape: [B, T, H, W].
        x = tf.placeholder('float', [None, inp_width, inp_height])
        m['x'] = x

        x_shape = tf.shape(x, name='x_shape')
        num_ex = x_shape[0: 1]
        num_ex_mul = tf.concat(0, [num_ex, tf.constant([1])])

        unroll_read_controller = _add_controller_rnn(
            model=m,
            timespan=timespan,
            inp_width=inp_width,
            inp_height=inp_height,
            ctl_inp_dim=hid_dec_dim,
            filter_size=filter_size_r,
            wd=wd,
            name='r'
        )
        lg_gamma_r = m['lg_gamma_r']
        filter_x_r = m['filter_x_r']
        filter_y_r = m['filter_y_r']

        # Read out image from the read head. Shape: T * [B, Fr, Fr].
        readout_x = [None] * timespan
        readout_err = [None] * timespan
        x_and_err = [None] * timespan
        # Read out image from the read head. Shape: T * [B, 2 * Fr * Fr].
        readout = [None] * timespan
        m['readout_x'] = readout_x

        # Write attention controller
        unroll_write_controller = _add_controller_rnn(
            model=m,
            timespan=timespan,
            inp_width=inp_width,
            inp_height=inp_height,
            ctl_inp_dim=hid_dec_dim,
            filter_size=filter_size_w,
            wd=wd,
            name='w'
        )
        lg_gamma_w = m['lg_gamma_w']
        filter_x_w = m['filter_x_w']
        filter_y_w = m['filter_y_w']

        # Write to canvas
        # Write out. Shape: T * [B, F, F].
        writeout = [None] * timespan
        # From decoder to write. Shape: [Hd, Fw * Fw].
        w_hdec_writeout = weight_variable(
            [hid_dec_dim, filter_size_w * filter_size_w], wd=wd,
            name='w_hdec_writeout')

        # Shape: [Fw * Fw].
        b_writeout = weight_variable(
            [filter_size_w * filter_size_w], wd=wd, name='b_writeout')
        m['w_hdec_writeout'] = w_hdec_writeout
        m['b_writeout'] = b_writeout

        # Add to canvas. Shape: T * [B, H, W].
        canvas_delta = [None] * timespan
        m['canvas_delta'] = canvas_delta

        # Canvas accumulating output. Shape: T * [B, H, W].
        canvas = [None] * (timespan + 1)
        w_canvas_init = weight_variable([inp_height, inp_width], wd=wd,
                                        name='w_canvas_init')
        canvas[-1] = w_canvas_init
        m['w_canvas_init'] = w_canvas_init

        # Reconstruction image.
        x_rec = [None] * timespan
        m['x_rec'] = x_rec

        # Error images (original substracted by drawn). Shape: T * [B, H, W].
        x_err = [None] * timespan

        # Encoder RNN
        w_henc_init = weight_variable([1, hid_enc_dim], wd=wd,
                                      name='w_henc_init')
        henc_init = tf.tile(w_henc_init, num_ex_mul, name='henc_init')
        unroll_encoder = _add_gated_rnn(
            model=m,
            timespan=timespan,
            inp_dim=2 * filter_size_r * filter_size_r + hid_dec_dim,
            hid_dim=hid_enc_dim,
            h_init=henc_init,
            wd=wd,
            name='enc'
        )
        h_enc = m['h_enc']
        m['w_henc_init'] = w_henc_init

        # Latent distribution
        if train:
            # Noise. Shape: [B, T, Hz]
            u = tf.placeholder('float', [None, timespan, hid_dim])
            u_l = tf.split(1, timespan, u)
            u_l_flat = [None] * timespan
            m['u'] = u

        # Mean of hidden latent variable. Shape: T * [B, Hz].
        mu_z = [None] * timespan
        # Hidden encoder to latent variable mean.
        w_henc_muz = weight_variable([hid_enc_dim, hid_dim], wd=wd,
                                     name='w_henc_muz')
        b_muz = weight_variable([hid_dim], wd=wd, name='w_henc_muz')
        m['w_henc_muz'] = w_henc_muz
        m['b_muz'] = b_muz

        # Hidden latent variable. Shape: T * [B, Hz].
        z = [None] * timespan

        if train:
            # Standard deviation of hidden latent variable. Shape: T * [B, Hz].
            lg_sigma_z = [None] * timespan
            sigma_z = [None] * timespan
            # Hidden encoder to latent variable std.
            w_henc_lgsigmaz = weight_variable(
                [hid_enc_dim, hid_dim], wd=wd, name='w_henc_lgsigmaz')
            b_lgsigmaz = weight_variable([hid_dim], wd=wd, name='b_lgsigmaz')
            m['w_henc_lgsigmaz'] = w_henc_lgsigmaz
            m['b_lgsigmaz'] = b_lgsigmaz

            # KL divergence
            kl_qzx_pz = [None] * timespan

        # Decoder RNN
        w_hdec_init = weight_variable([1, hid_dec_dim], wd=wd,
                                      name='w_hdec_init')
        hdec_init = tf.tile(w_hdec_init, num_ex_mul, name='hdec_init')
        m['w_hdec_init'] = w_hdec_init
        unroll_decoder = _add_gated_rnn(
            model=m,
            timespan=timespan,
            inp_dim=hid_dim,
            hid_dim=hid_dec_dim,
            h_init=hdec_init,
            name='dec'
        )
        h_dec = m['h_dec']

        #########################################################
        # Computation graph
        #########################################################
        for t in xrange(timespan):
            # [B, H * W]
            x_err[t] = tf.sub(x, tf.sigmoid(canvas[t - 1]),
                              name='x_err_{}'.format(t))

            if train:
                u_l_flat[t] = tf.reshape(u_l[t], [-1, hid_dim],
                                         name='u_flat_{}'.format(t))

            # Read attention selector
            unroll_read_controller(ctl_inp=h_dec[t - 1], time=t)

            # [B, 1, 1] * [B, F, H] * [B, H, W] * [B, W, F] = [B, F, F]
            readout_x[t] = tf.mul(tf.exp(lg_gamma_r[t]), _batch_matmul(
                _batch_matmul(filter_y_r[t], x, adj_x=True, rep_x=False),
                filter_x_r[t]),
                name='readout_x_{}'.format(t))
            readout_err[t] = tf.mul(tf.exp(lg_gamma_r[t]), _batch_matmul(
                _batch_matmul(filter_y_r[t], x_err[t],
                              adj_x=True, rep_x=False),
                filter_x_r[t]),
                name='readout_err_{}'.format(t))

            # [B, 2 * F * F]
            x_and_err[t] = [tf.reshape(readout_x[t],
                                       [-1, filter_size_r * filter_size_r]),
                            tf.reshape(readout_err[t],
                                       [-1, filter_size_r * filter_size_r])]
            readout[t] = tf.concat(1, x_and_err[t],
                                   name='readout_{}'.format(t))

            # Encoder RNN
            enc_inp = tf.concat(1,
                                [readout[t], h_dec[t - 1]],
                                name='enc_inp_{}'.format(t))
            unroll_encoder(inp=enc_inp, time=t)

            # Latent distribution
            # [B, He] * [He, H]
            mu_z[t] = tf.add(tf.matmul(h_enc[t], w_henc_muz), b_muz,
                             name='mu_z_{}'.format(t))
            if train:
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
                                      name='kl_qzx_pz_{}'.format(t))
            else:
                z[t] = mu_z[t]

            # Decoder RNN
            unroll_decoder(inp=z[t], time=t)

            # Write to canvas
            unroll_write_controller(ctl_inp=h_dec[t], time=t)

            # [B, F, Fw]
            writeout[t] = tf.reshape(tf.matmul(h_dec[t], w_hdec_writeout) +
                                     b_writeout,
                                     [-1, filter_size_w, filter_size_w],
                                     name='writeout_{}'.format(t))

            # [B, H, Fw] * [B, Fw, Fw] * [B, Fw, W] = [B, H, W]
            canvas_delta[t] = tf.mul(1 / tf.exp(lg_gamma_w[t]),
                                     _batch_matmul(_batch_matmul(
                                         filter_y_w[t], writeout[t],
                                         rep_x=False),
                                     filter_x_w[t], adj_y=True),
                                     name='canvas_delta_{}'.format(t))
            # [B, H, W]
            canvas[t] = canvas[t - 1] + canvas_delta[t]
            x_rec[t] = tf.sigmoid(canvas[t], name='x_rec')

        #########################################################
        # Loss and gradient
        #########################################################
        if train:
            eps = 1e-7
            kl_qzx_pz_sum = tf.reduce_sum(tf.pack(kl_qzx_pz))
            log_pxz_sum = tf.reduce_sum(x * tf.log(x_rec[-1] + eps) +
                                        (1 - x) * tf.log(1 - x_rec[-1] + eps),
                                        name='ce_sum')
            # w_kl = opt['w_kl']
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
            total_loss = tf.add_n(tf.get_collection('losses'),
                                  name='total_loss')

            lr = 1e-4
            train_step = GradientClipOptimizer(
                tf.train.AdamOptimizer(lr, epsilon=eps), clip=1.0).minimize(
                total_loss)
            m['ce'] = ce
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
    parser.add_argument('-logs', default=None,
                        help='Training curve logs folder')
    parser.add_argument('-localhost', default='localhost',
                        help='Local domain name')
    parser.add_argument('-restore', default=None,
                        help='Model save folder to restore from')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    parser.add_argument('-seed', default=100, type=int,
                        help='Training seed')
    parser.add_argument('-filter_size_r', default=2, type=int,
                        help='Read filter size')
    parser.add_argument('-filter_size_w', default=5, type=int,
                        help='Write filter size')
    parser.add_argument('-w_kl', default=1, type=float,
                        help='Mixing ratio of KL divergence')
    
    args = parser.parse_args()

    return args


def preprocess(x, opt):
    if opt['output_dist'] == 'Bernoulli':
        return (x > 0.5).astype('float32').reshape([-1, 28, 28])
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

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)

    if args.restore:
        log.info('Restoring from {}'.format(args.restore))
        opt_fname = os.path.join(args.restore, 'opt.pkl')

        # Load model configs.
        with open(opt_fname, 'rb') as f_opt:
            opt = pkl.load(f_opt)
        log.info(opt)

        ckpt_fname, latest_step = _get_latest_ckpt(args.restore)
        step = latest_step
        log.info('Step {}'.format(step))
    else:
        log.info('Initializing new model')
        opt = {
            'inp_height': 28,
            'inp_width': 28,
            'timespan': 64,
            'filter_size_r': args.filter_size_r,
            'filter_size_w': args.filter_size_w,
            'hid_enc_dim': 256,
            'hid_dim': 100,
            'hid_dec_dim': 256,
            'weight_decay': 5e-5,
            'output_dist': 'Bernoulli',
            'squash': False,
            'w_kl': args.w_kl
        }
        step = 0

    m = get_model(opt, device=device)
    sess = tf.Session()
    saver = tf.train.Saver(tf.all_variables())

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    task_name = 'draw_mnist'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    results_folder = args.results
    logs_folder = args.logs
    exp_folder = os.path.join(results_folder, model_id)

    # Create time series logger
    if args.logs:
        exp_logs_folder = os.path.join(logs_folder, model_id)
        train_ce_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'train_ce.csv'), 'train_ce',
            buffer_size=25)
        valid_ce_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'valid_ce.csv'), 'valid_ce',
            buffer_size=2)
        step_time_logger = TimeSeriesLogger(
            os.path.join(exp_logs_folder, 'step_time.csv'), 'step time (ms)',
            buffer_size=25)
        log.info(
            'Curves can be viewed at: http://{}/visualizer?id={}'.format(
                args.localhost, model_id))

    random = np.random.RandomState(args.seed)

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

        if args.logs:
            valid_ce_logger.add(step, valid_ce)

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x = preprocess(batch[0], opt)
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['hid_dim']])
            tim = time.time()
            r = sess.run([m['ce'], m['train_step']], feed_dict={
                m['x']: x,
                m['u']: u
            })
            if step % 10 == 0:
                ce = r[0]
                step_time = (time.time() - tim) * 1000
                log.info('{:d} train ce {:.4f} t {:.2f}ms'.format(
                    step, ce, step_time))
                if args.logs:
                    train_ce_logger.add(step, ce)
                    step_time_logger.add(step, step_time)

            step += 1

            # Save model
            if step % args.steps_per_ckpt == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)

    sess.close()
    pass
