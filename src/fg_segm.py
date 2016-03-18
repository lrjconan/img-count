"""
Foreground segmentation used to pre-train box proposal network.

Usage: python rec_ins_segm_attn.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
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
from utils import plot_utils

import fg_segm_models as models

log = logger.get()


def get_dataset(opt):
    if os.path.exists('/u/mren'):
        dataset_folder = '/ais/gobi3/u/mren/data/kitti'
    else:
        dataset_folder = '/home/mren/data/kitti'
    opt['timespan'] = 20
    opt['num_examples'] = -1
    dataset = {}
    dataset['train'] = kitti.get_foreground_dataset(dataset_folder, opt, split='train')
    dataset['valid'] = kitti.get_foreground_dataset(dataset_folder, opt, split='valid')

    return dataset


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

    args = parser.parse_args()

    return args


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


def _get_ts_loggers(model_opt, debug_bn=False):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1)
    loggers['iou'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'iou.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='IoU',
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
    model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
    saver.save_opt(model_opt_fname, model_opt)
    log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

    pass


def _get_max_items_per_row(inp_height, inp_width):
    return 5


def _get_num_batch_valid(dataset_name):
    return 10


def _get_batch_fn(dataset):
    """
    Preprocess mini-batch data given start and end indices.
    """
    def get_batch(idx):
        x_bat = dataset['input'][idx]
        y_bat = dataset['label'][idx]
        x_bat, y_bat = preprocess(x_bat, y_bat)

        return x_bat, y_bat

    return get_batch


def _run_model(m, names, feed_dict):
    symbol_list = [m[r] for r in names]
    results = sess.run(symbol_list, feed_dict=feed_dict)
    results_dict = {}
    for rr, name in zip(results, names):
        results_dict[name] = rr

    return results_dict


def preprocess(inp, label):
    """Preprocess training data."""
    ls = label.shape
    return (inp.astype('float32') / 255,
            label.astype('float32').reshape(ls[0], ls[1], ls[2], 1))

if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)
    saver = None
    train_opt = _make_train_opt(args)

    model_opt = {
        'inp_height': 128,
        'inp_width': 448,
        'inp_depth': 3,
        'padding': 16,
        'cnn_filter_size': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'cnn_depth': [4, 4, 8, 8, 16, 16, 32, 32, 64, 64],
        'cnn_pool': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'dcnn_filter_size': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'dcnn_depth': [64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 1],
        'dcnn_pool': [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
        'weight_decay': 5e-5,
        'use_bn': True,
        'rnd_hflip': True,
        'rnd_vflip': False,
        'rnd_transpose': False,
        'rnd_colour': False,
        'base_learn_rate': 1e-3,
        'learn_rate_decay': 0.96,
        'steps_per_learn_rate_decay': 5000,
    }
    data_opt = {
        'height': model_opt['inp_height'],
        'width': model_opt['inp_width'],
        'timespan': 20,
        'num_train': -1,
        'num_valid': -1,
        'has_valid': True
    }

    model_id = _get_model_id('rec_ins_segm')
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
    m = models.get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = get_dataset(data_opt)

    sess = tf.Session()

    # Create time series loggers
    loggers = {}
    if train_opt['logs']:
        loggers = _get_ts_loggers(model_opt, debug_bn=train_opt['debug_bn'])
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
        def _run_samples(x, y, phase_train, fname_input, fname_output):
            _outputs = ['x_trans', 'y_gt_trans', 'y_out']
            _max_items = _get_max_items_per_row(x.shape[1], x.shape[2])

            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y}
            _r = _run_model(m, _outputs, _feed_dict)

            pu.plot_thumbnails(fname_input, _r['x_trans'],
                               max_items_per_row=_max_items)

            pu.plot_thumbnails(fname_output, _r['y_out'].reshape(
                y.shape[0], y.shape[1], y.shape[2], 1), axis=3,
                max_items_per_row=_max_items)

            pass

    def get_outputs_valid():
        _outputs = ['loss', 'iou_soft', 'iou_hard']

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'iou_soft', 'iou_hard']

        return _outputs

    def run_stats(step, num_batch, batch_iter, outputs, write_log, phase_train):
        """Validation"""
        nvalid = num_batch * batch_size
        r = {}

        for bb in xrange(num_batch):
            _x, _y = batch_iter.next()
            _feed_dict = {m['x']: _x, m['phase_train']: phase_train,
                          m['y_gt']: _y}
            _r = _run_model(m, outputs, _feed_dict)
            bat_sz = _x.shape[0]

            for key in _r.iterkeys():
                if key in r:
                    r[key] += _r[key] * bat_sz / nvalid
                else:
                    r[key] = _r[key] * bat_sz / nvalid

        log.info('{:d} loss {:.4f}'.format(step, r['loss']))
        write_log(loggers, r)

        pass

    def write_log_valid(loggers, r):
        loggers['loss'].add(step, ['', r['loss']])
        loggers['iou'].add(step, ['', r['iou_soft'], '', r['iou_hard']])

        pass

    def write_log_trainval(loggers, r, bn=False):
        loggers['loss'].add(step, [r['loss'], ''])
        loggers['iou'].add(step, [r['iou_soft'], '', r['iou_hard'], ''])

        pass

    def train_step(step, x, y):
        """Train step"""
        _outputs = ['loss', 'train_step']
        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y}
        _start_time = time.time()
        r = _run_model(m, _outputs, _feed_dict)
        _step_time = (time.time() - _start_time) * 1000

        # Print statistics
        if step % train_opt['steps_per_log'] == 0:
            log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, r['loss'],
                                                          _step_time))
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['step_time'].add(step, _step_time)

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
        num_batch_valid = _get_num_batch_valid(args.dataset)
        batch_iter_trainval = BatchIterator(num_ex_train,
                                            batch_size=batch_size,
                                            get_fn=get_batch_train,
                                            cycle=True,
                                            progress_bar=False)
        outputs_trainval = get_outputs_trainval()
        if train_opt['debug_bn']:
            if train_opt['has_valid']:
                outputs_valid.extend(get_outputs_bn())
            outputs_trainval.extend(get_outputs_bn())

        for _x, _y in BatchIterator(num_ex_train,
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
            train_step(step, _x, _y)

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
