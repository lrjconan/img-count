"""
Box proposal network

Usage: python ris_box.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import datetime
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle as pkl
import sys
import tensorflow as tf
import time

from data_api import synth_shape
from data_api import cvppp
from data_api import kitti

from utils import log_manager
from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger
from utils import plot_utils as pu

import ris_base as base
import ris_box_model as model

log = logger.get()

kKittiNumObj = 19
kKittiInpHeight = 128
kKittiInpWidth = 448


def get_dataset(opt):
    if os.path.exists('/u/mren'):
        dataset_folder = '/ais/gobi3/u/mren/data/kitti/object'
    else:
        dataset_folder = '/home/mren/data/kitti'
    opt['timespan'] = 20
    opt['num_examples'] = -1
    dataset = {}
    dataset['train'] = kitti.get_dataset(dataset_folder, opt,
                                         split='train')
    dataset['valid'] = kitti.get_dataset(dataset_folder, opt,
                                         split='valid')

    return dataset


def plot_output(fname, y_out, s_out, match, attn=None, max_items_per_row=9):
    """Plot some test samples.

    Args:
        fname: str, image output filename.
        y_out: [B, T, H, W, D], segmentation output of the model.
        s_out: [B, T], confidence score output of the model.
        match: [B, T, T], matching matrix.
        attn: ([B, T, 2], [B, T, 2]), top left and bottom right coordinates of
        the attention box.
    """
    num_ex = y_out.shape[0]
    num_items = y_out.shape[1]
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    cmap = ['r', 'y', 'c', 'g', 'm']

    if attn:
        attn_top_left_y = attn[0][:, :, 0]
        attn_top_left_x = attn[0][:, :, 1]
        attn_bot_right_y = attn[1][:, :, 0]
        attn_bot_right_x = attn[1][:, :, 1]

    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            axarr[row, col].imshow(y_out[ii, jj])
            matched = match[ii, jj].nonzero()[0]
            axarr[row, col].text(0, 0, '{:.2f} {}'.format(
                s_out[ii, jj], matched),
                color=(0, 0, 0), size=8)

            if attn:
                # Plot attention box.
                axarr[row, col].add_patch(patches.Rectangle(
                    (attn_top_left_x[ii, jj], attn_top_left_y[ii, jj]),
                    attn_bot_right_x[ii, jj] - attn_top_left_x[ii, jj],
                    attn_bot_right_y[ii, jj] - attn_top_left_y[ii, jj],
                    fill=False,
                    color='g'))

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')

    pass


def _add_model_args(parser):
    # Default model options
    kCtrlCnnFilterSize = '3,3,3,3,3,3,3,3,3,3'
    kCtrlCnnDepth = '4,4,8,8,16,16,32,32,64,64'
    kCtrlCnnPool = '1,2,1,2,1,2,1,2,1,2'
    kBoxLossFn = 'iou'
    kCtrlMlpDim = 256
    kNumCtrlMlpLayers = 2
    kCtrlRnnHiddenDim = 256
    kLearnRateDecay = 0.96
    kStepsPerLearnRateDecay = 5000
    kFilterHeight = 48
    kFilterWidth = 48
    kCtrlRnnInpStruct = 'dense'
    kNumCtrlRNNIter = 5
    kNumGlimpseMlpLayers = 2

    parser.add_argument('-filter_height', default=kFilterHeight, type=int,
                        help='Attention filter width')
    parser.add_argument('-filter_width', default=kFilterWidth, type=int,
                        help='Attention filter size')
    parser.add_argument('-ctrl_cnn_filter_size', default=kCtrlCnnFilterSize,
                        help='Comma delimited integers')
    parser.add_argument('-ctrl_cnn_depth', default=kCtrlCnnDepth,
                        help='Comma delimited integers')
    parser.add_argument('-ctrl_cnn_pool', default=kCtrlCnnPool,
                        help='Comma delimited integers')
    parser.add_argument('-box_loss_fn', default=kBoxLossFn,
                        help='Loss function for box regressor')
    parser.add_argument('-fixed_order', action='store_true',
                        help='Train with fixed order labels')
    parser.add_argument('-pretrain_cnn', default=None,
                        help='Pretrained CNN weights')
    parser.add_argument('-ctrl_rnn_hid_dim', default=kCtrlRnnHiddenDim,
                        type=int, help='RNN hidden dimension')
    parser.add_argument('-num_ctrl_mlp_layers', default=kNumCtrlMlpLayers,
                        type=int, help='Number of controller MLP layers')
    parser.add_argument('-ctrl_mlp_dim', default=kCtrlMlpDim,
                        type=int, help='Controller MLP dimension')
    parser.add_argument('-use_iou_box', action='store_true',
                        help='Use hard box IOU')
    parser.add_argument('-learn_rate_decay', default=kLearnRateDecay,
                        type=float, help='Model learning rate decay')
    parser.add_argument('-steps_per_learn_rate_decay',
                        default=kStepsPerLearnRateDecay, type=int,
                        help='Steps every learning rate decay')
    parser.add_argument('-squash_ctrl_params', action='store_true',
                        help='Whether to squash control parameters.')
    parser.add_argument('-ctrl_rnn_inp_struct', default=kCtrlRnnInpStruct,
                        help='Dense or attn')
    parser.add_argument('-num_ctrl_rnn_iter', default=kNumCtrlRNNIter,
                        type=int, help='Number of control RNN iterations')
    parser.add_argument('-num_glimpse_mlp_layers', default=kNumGlimpseMlpLayers,
                        type=int, help='Number of glimpse MLP layers')

    pass


def _add_training_args(parser):
    # Default training options
    kNumSteps = 500000
    kStepsPerCkpt = 1000
    kStepsPerValid = 250
    kStepsPerTrainval = 100
    kStepsPerPlot = 50
    kStepsPerLog = 20
    kBatchSize = 32

    # Training options
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-steps_per_valid', default=kStepsPerValid,
                        type=int, help='Number of steps per validation')
    parser.add_argument('-steps_per_trainval', default=kStepsPerTrainval,
                        type=int, help='Number of steps per train validation')
    parser.add_argument('-steps_per_plot', default=kStepsPerPlot,
                        type=int, help='Number of steps per plot samples')
    parser.add_argument('-steps_per_log', default=kStepsPerLog,
                        type=int, help='Number of steps per log')
    parser.add_argument('-batch_size', default=kBatchSize,
                        type=int, help='Size of a mini-batch')
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
    parser.add_argument('-num_samples_plot', default=10, type=int,
                        help='Number of samples to plot')
    parser.add_argument('-save_ckpt', action='store_true',
                        help='Whether to store checkpoints')
    parser.add_argument('-no_valid', action='store_true',
                        help='Use the whole training set.')

    pass


def _parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Foreground Segmentation')

    _add_training_args(parser)
    _add_model_args(parser)

    args = parser.parse_args()

    return args


def _make_model_opt(args):
    ccnn_fsize_list = args.ctrl_cnn_filter_size.split(',')
    ccnn_fsize_list = [int(fsize) for fsize in ccnn_fsize_list]
    ccnn_depth_list = args.ctrl_cnn_depth.split(',')
    ccnn_depth_list = [int(depth) for depth in ccnn_depth_list]
    ccnn_pool_list = args.ctrl_cnn_pool.split(',')
    ccnn_pool_list = [int(pool) for pool in ccnn_pool_list]

    timespan = kKittiNumObj + 1
    inp_height = kKittiInpHeight
    inp_width = kKittiInpWidth

    model_opt = {
        'timespan': timespan,
        'inp_height': inp_height,
        'inp_width': inp_width,
        'inp_depth': 3,
        'padding': 16,
        'filter_height': args.filter_height,
        'filter_width': args.filter_width,
        'ctrl_cnn_filter_size': ccnn_fsize_list,
        'ctrl_cnn_depth': ccnn_depth_list,
        'ctrl_cnn_pool': ccnn_pool_list,
        'ctrl_rnn_hid_dim': args.ctrl_rnn_hid_dim,
        'num_ctrl_mlp_layers': args.num_ctrl_mlp_layers,
        'ctrl_mlp_dim': args.ctrl_mlp_dim,
        'attn_box_padding_ratio': 0.2,
        'weight_decay': 5e-5,
        'use_bn': True,
        'box_loss_fn': args.box_loss_fn,
        'base_learn_rate': 1e-3,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
        # 'gt_selector': 'greedy_match',
        'gt_selector': 'greedy',
        'fixed_order': args.fixed_order,
        'pretrain_cnn': args.pretrain_cnn,
        'use_iou_box': args.use_iou_box,
        'squash_ctrl_params': args.squash_ctrl_params,
        'ctrl_rnn_inp_struct': args.ctrl_rnn_inp_struct,
        'num_ctrl_rnn_iter': args.num_ctrl_rnn_iter,
        'num_glimpse_mlp_layers': args.num_glimpse_mlp_layers,

        'rnd_hflip': True,
        'rnd_vflip': False,
        'rnd_transpose': False,
        'rnd_colour': False
    }

    return model_opt


def _make_data_opt(args):
    data_opt = {
        'height': 128,
        'width': 448,
        'timespan': 20,
        'num_train': -1,
        'num_valid': -1,
        'has_valid': True
    }

    return data_opt


def _make_train_opt(args):
    """Train opt"""
    train_opt = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt,
        'steps_per_valid': args.steps_per_valid,
        'steps_per_trainval': args.steps_per_trainval,
        'steps_per_plot': args.steps_per_plot,
        'steps_per_log': args.steps_per_log,
        'has_valid': not args.no_valid,
        'results': args.results,
        'restore': args.restore,
        'save_ckpt': args.save_ckpt,
        'logs': args.logs,
        'gpu': args.gpu,
        'localhost': args.localhost
    }

    return train_opt


def _get_model_id(task_name):
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    return model_id


def _get_ts_loggers(model_opt):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1)
    loggers['box_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'box_loss.csv'), ['train', 'valid'],
        name='Box Loss',
        buffer_size=1)
    loggers['conf_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'conf_loss.csv'), ['train', 'valid'],
        name='Confidence Loss',
        buffer_size=1)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
        name='Step time',
        buffer_size=1)

    return loggers


def _get_plot_loggers(model_opt, train_opt):
    samples = {}
    _ssets = ['train', 'valid']
    for _set in _ssets:
        labels = ['input', 'output']
        for name in labels:
            key = '{}_{}'.format(name, _set)
            samples[key] = LazyRegisterer(
                os.path.join(logs_folder, '{}.png'.format(key)),
                'image', 'Samples {} {}'.format(name, _set))

    return samples


def _register_raw_logs(log_manager, log, model_opt, saver):
    log_manager.register(log.filename, 'plain', 'Raw logs')
    cmd_fname = os.path.join(logs_folder, 'cmd.txt')
    with open(cmd_fname, 'w') as f:
        f.write(' '.join(sys.argv))
    log_manager.register(cmd_fname, 'plain', 'Command-line arguments')
    model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
    saver.save_opt(model_opt_fname, model_opt)
    log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

    pass


def _get_max_items_per_row(inp_height, inp_width):
    return 5


def _get_num_batch_valid():
    return 10


def _get_batch_fn(dataset):
    """
    Preprocess mini-batch data given start and end indices.
    """
    def get_batch(idx):
        x_bat = dataset['input'][idx]
        y_bat = dataset['label_segmentation'][idx]
        s_bat = dataset['label_score'][idx]
        x_bat, y_bat, s_bat = preprocess(x_bat, y_bat, s_bat)

        return x_bat, y_bat, s_bat

    return get_batch


def _run_model(m, names, feed_dict):
    symbol_list = [m[r] for r in names]
    results = sess.run(symbol_list, feed_dict=feed_dict)
    results_dict = {}
    for rr, name in zip(results, names):
        results_dict[name] = rr

    return results_dict


def preprocess(inp, label_segmentation, label_score):
    """Preprocess training data."""
    return (inp.astype('float32') / 255,
            label_segmentation.astype('float32'),
            label_score.astype('float32'))


if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)
    saver = None
    train_opt = _make_train_opt(args)
    model_opt = _make_model_opt(args)
    data_opt = _make_data_opt(args)

    # Restore previously saved checkpoints.
    if train_opt['restore']:
        saver = Saver(train_opt['restore'])
        ckpt_info = saver.get_ckpt_info()
        model_opt = ckpt_info['model_opt']
        data_opt = ckpt_info['data_opt']
        ckpt_fname = ckpt_info['ckpt_fname']
        step = ckpt_info['step']
        model_id = ckpt_info['model_id']
        exp_folder = train_opt['restore']
    else:
        model_id = _get_model_id('ris_box')
        step = 0
        exp_folder = os.path.join(train_opt['results'], model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    if not train_opt['save_ckpt']:
        log.warning(
            'Checkpoints saving is turned off. Use -save_ckpt flag to save.')

    # Logger
    if train_opt['logs']:
        logs_folder = train_opt['logs']
        logs_folder = os.path.join(logs_folder, model_id)
        log = logger.get(os.path.join(logs_folder, 'raw'))
    else:
        log = logger.get()

    # Log arguments
    log.log_args()

    # Set device
    if train_opt['gpu'] >= 0:
        device = '/gpu:{}'.format(train_opt['gpu'])
    else:
        device = '/cpu:0'

    # Train loop options
    log.info('Building model')
    m = model.get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = get_dataset(data_opt)
    if model_opt['fixed_order']:
        dataset['train']['label_segmentation'] = base.sort_by_segm_size(
            dataset['train']['label_segmentation'])
        dataset['valid']['label_segmentation'] = base.sort_by_segm_size(
            dataset['valid']['label_segmentation'])

    sess = tf.Session()

    # Restore/intialize weights
    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series loggers
    if train_opt['logs']:
        loggers = _get_ts_loggers(model_opt)
        _register_raw_logs(log_manager, log, model_opt, saver)
        samples = _get_plot_loggers(model_opt, train_opt)
        _log_url = 'http://{}/deep-dashboard?id={}'.format(
            train_opt['localhost'], model_id)
        log.info('Visualization can be viewed at: {}'.format(_log_url))

    batch_size = args.batch_size
    log.info('Batch size: {}'.format(batch_size))
    num_ex_train = dataset['train']['input'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    log.info('Number of training examples: {}'.format(num_ex_train))

    if train_opt['has_valid']:
        num_ex_valid = dataset['valid']['input'].shape[0]
        get_batch_valid = _get_batch_fn(dataset['valid'])
        log.info('Number of validation examples: {}'.format(num_ex_valid))

    def run_samples():
        """Samples"""
        def _run_samples(x, y, s, phase_train, fname_input, fname_output):
            _outputs = ['x_trans', 'y_gt_trans', 'attn_top_left',
                        'attn_bot_right', 'attn_top_left_gt',
                        'attn_bot_right_gt', 'match_box', 's_out']
            _max_items = _get_max_items_per_row(x.shape[1], x.shape[2])

            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y, m['s_gt']: s}
            _r = _run_model(m, _outputs, _feed_dict)

            _x = np.expand_dims(_r['x_trans'], 1)
            _x = np.tile(_x, [1, y.shape[1], 1, 1, 1])

            plot_output(fname_input, _x,
                        s_out=s,
                        match=_r['match_box'],
                        attn=(_r['attn_top_left_gt'], _r['attn_bot_right_gt']),
                        max_items_per_row=_max_items)

            plot_output(fname_output, _x,
                        s_out=_r['s_out'],
                        match=_r['match_box'],
                        attn=(_r['attn_top_left'], _r['attn_bot_right']),
                        max_items_per_row=_max_items)

            pass

        # Plot some samples.
        _ssets = ['train']
        if train_opt['has_valid']:
            _ssets.append('valid')

        for _set in _ssets:
            _is_train = _set == 'train'
            _get_batch = get_batch_train if _is_train else get_batch_valid
            _num_ex = num_ex_train if _is_train else num_ex_valid
            log.info('Plotting {} samples'.format(_set))
            _x, _y, _s = _get_batch(
                np.arange(min(_num_ex, args.num_samples_plot)))

            _run_samples(
                _x, _y, _s, _is_train,
                fname_input=samples['input_{}'.format(_set)].get_fname(),
                fname_output=samples['output_{}'.format(_set)].get_fname())

            if not samples['output_{}'.format(_set)].is_registered():
                for _name in ['input', 'output']:
                    samples['{}_{}'.format(_name, _set)].register()

        pass

    def get_outputs_valid():
        _outputs = ['loss', 'box_loss', 'conf_loss']

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'box_loss', 'conf_loss']

        return _outputs

    def run_stats(step, num_batch, batch_iter, outputs, write_log, phase_train):
        """Validation"""
        nvalid = num_batch * batch_size
        r = {}

        for bb in xrange(num_batch):
            _x, _y, _s = batch_iter.next()
            _feed_dict = {m['x']: _x, m['phase_train']: phase_train,
                          m['y_gt']: _y, m['s_gt']: _s}
            _r = _run_model(m, outputs, _feed_dict)
            bat_sz = _x.shape[0]

            for key in _r.iterkeys():
                if key in r:
                    r[key] += _r[key] * bat_sz / nvalid
                else:
                    r[key] = _r[key] * bat_sz / nvalid

        log.info('{:d} loss {:.4f}'.format(step, r['loss']))
        write_log(step, loggers, r)

        pass

    def write_log_valid(step, loggers, r):
        loggers['loss'].add(step, ['', r['loss']])
        loggers['box_loss'].add(step, ['', r['box_loss']])
        loggers['conf_loss'].add(step, ['', r['conf_loss']])

        pass

    def write_log_trainval(step, loggers, r, bn=False):
        loggers['loss'].add(step, [r['loss'], ''])
        loggers['box_loss'].add(step, [r['box_loss'], ''])
        loggers['conf_loss'].add(step, [r['conf_loss'], ''])

        pass

    def train_step(step, x, y, s):
        """Train step"""
        _outputs = ['loss', 'train_step']
        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                      m['s_gt']: s}
        _start_time = time.time()
        r = _run_model(m, _outputs, _feed_dict)
        _step_time = (time.time() - _start_time) * 1000

        # Print statistics
        if step % train_opt['steps_per_log'] == 0:
            log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, r['loss'],
                                                          _step_time))
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['step_time'].add(step, _step_time)

        # Check NaN.
        if np.isnan(r['loss']):
            log.error('NaN occurred. Saving last step.')
            saver.save(sess, global_step=step)
            input_file = h5py.File(os.path.join(exp_folder, 'nan_input.h5'))
            input_file['x'] = x
            input_file['y'] = y
            input_file['s'] = s
            raise Exception('NaN')

        pass

    def train_loop(step=0):
        """Train loop"""
        if train_opt['has_valid']:
            batch_iter_valid = BatchIterator(num_ex_valid,
                                             batch_size=batch_size,
                                             get_fn=get_batch_valid,
                                             cycle=True,
                                             progress_bar=False)
            outputs_valid = get_outputs_valid()
        num_batch_valid = _get_num_batch_valid()
        batch_iter_trainval = BatchIterator(num_ex_train,
                                            batch_size=batch_size,
                                            get_fn=get_batch_train,
                                            cycle=True,
                                            progress_bar=False)
        outputs_trainval = get_outputs_trainval()

        for _x, _y, _s in BatchIterator(num_ex_train,
                                        batch_size=batch_size,
                                        get_fn=get_batch_train,
                                        cycle=True,
                                        progress_bar=False):
            # Run validation stats
            if train_opt['has_valid']:
                if step % train_opt['steps_per_valid'] == 0:
                    log.info('Running validation')
                    run_stats(step, num_batch_valid, batch_iter_valid,
                              outputs_valid, write_log_valid, False)
                    pass

            # Train stats
            if step % train_opt['steps_per_trainval'] == 0:
                log.info('Running train validation')
                run_stats(step, num_batch_valid, batch_iter_trainval,
                          outputs_trainval, write_log_trainval, True)
                pass

            # Plot samples
            if step % train_opt['steps_per_plot'] == 0:
                run_samples()
                pass

            # Train step
            train_step(step, _x, _y, _s)

            # Model ID reminder
            if step % (10 * train_opt['steps_per_log']) == 0:
                log.info('model id {}'.format(model_id))
                pass

            # Save model
            if args.save_ckpt and step % train_opt['steps_per_ckpt'] == 0:
                saver.save(sess, global_step=step)
                pass

            step += 1

            # Termination
            if step > train_opt['num_steps']:
                break

        pass

    train_loop(step=step)

    sess.close()
    for logger in loggers.itervalues():
        logger.close()
