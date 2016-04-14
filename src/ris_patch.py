"""
Train a patch net. Usage: python ris_patch.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import datetime
import h5py
import numpy as np
import os
import tensorflow as tf
import time

from utils.log_manager import LogManager
from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger
from utils import plot_utils as pu

import ris_patch_model as patch_model
import ris_train_base as trainer

log = logger.get()


def add_model_args(parser):
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--base_learn_rate', default=1e-3, type=float)
    parser.add_argument('--learn_rate_decay', default=0.96, type=float)
    parser.add_argument('--steps_per_learn_rate_decay', default=5000, type=int)
    parser.add_argument('--segm_loss_fn', default='iou')
    parser.add_argument('--mlp_dropout', default=None, type=float)
    parser.add_argument('--no_cum_min', action='store_true')
    parser.add_argument('--fixed_order', action='store_true')
    parser.add_argument('--padding', default=16, type=int)
    parser.add_argument('--filter_height', default=48, type=int)
    parser.add_argument('--filter_width', default=48, type=int)
    parser.add_argument('--attn_cnn_filter_size', default='3,3,3')
    parser.add_argument('--attn_cnn_depth', default='4,8,16')
    parser.add_argument('--attn_cnn_pool', default='2,2,2')
    parser.add_argument('--attn_dcnn_filter_size', default='3,3,3,3')
    parser.add_argument('--attn_dcnn_depth', default='16,8,4,1')
    parser.add_argument('--attn_dcnn_pool', default='2,2,2,1')
    parser.add_argument('--num_attn_mlp_layers', default=1, type=int)
    parser.add_argument('--attn_mlp_depth', default=6, type=int)
    parser.add_argument('--attn_box_padding_ratio', default=0.2, type=float)
    parser.add_argument('--gt_box_ctr_noise', default=0.05, type=float)
    parser.add_argument('--gt_box_pad_noise', default=0.1, type=float)
    parser.add_argument('--gt_segm_noise', default=0.3, type=float)
    parser.add_argument('--clip_gradient', default=1.0, type=float)

    pass


def make_model_opt(args):
    """Convert command-line arguments into model opt dict."""
    inp_height, inp_width, timespan = trainer.get_inp_dim(args.dataset)
    rnd_hflip, rnd_vflip, rnd_transpose, rnd_colour = \
        trainer.get_inp_transform(args.dataset)

    if args.dataset == 'synth_shape':
        timespan = args.max_num_objects + 1

    acnn_fsize_list = args.attn_cnn_filter_size.split(',')
    acnn_fsize_list = [int(fsize) for fsize in acnn_fsize_list]
    acnn_depth_list = args.attn_cnn_depth.split(',')
    acnn_depth_list = [int(depth) for depth in acnn_depth_list]
    acnn_pool_list = args.attn_cnn_pool.split(',')
    acnn_pool_list = [int(pool) for pool in acnn_pool_list]

    attn_dcnn_fsize_list = args.attn_dcnn_filter_size.split(',')
    attn_dcnn_fsize_list = [int(fsize) for fsize in attn_dcnn_fsize_list]
    attn_dcnn_depth_list = args.attn_dcnn_depth.split(',')
    attn_dcnn_depth_list = [int(depth) for depth in attn_dcnn_depth_list]
    attn_dcnn_pool_list = args.attn_dcnn_pool.split(',')
    attn_dcnn_pool_list = [int(pool) for pool in attn_dcnn_pool_list]

    model_opt = {
        'inp_height': inp_height,
        'inp_width': inp_width,
        'inp_depth': 3,
        'padding': args.padding,
        'filter_height': args.filter_height,
        'filter_width': args.filter_width,
        'timespan': timespan,

        'attn_cnn_filter_size': acnn_fsize_list,
        'attn_cnn_depth': acnn_depth_list,
        'attn_cnn_pool': acnn_pool_list,

        'attn_dcnn_filter_size': attn_dcnn_fsize_list,
        'attn_dcnn_depth': attn_dcnn_depth_list,
        'attn_dcnn_pool': attn_dcnn_pool_list,

        'attn_mlp_depth': args.attn_mlp_depth,
        'num_attn_mlp_layers': args.num_attn_mlp_layers,
        'mlp_dropout': args.mlp_dropout,

        'weight_decay': args.weight_decay,
        'base_learn_rate': args.base_learn_rate,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,

        'segm_loss_fn': args.segm_loss_fn,
        'use_bn': True,
        'attn_box_padding_ratio': args.attn_box_padding_ratio,
        'gt_box_ctr_noise': args.gt_box_ctr_noise,
        'gt_box_pad_noise': args.gt_box_pad_noise,
        'gt_segm_noise': args.gt_segm_noise,
        'clip_gradient': args.clip_gradient,
        'fixed_order': args.fixed_order,

        'rnd_hflip': rnd_hflip,
        'rnd_vflip': rnd_vflip,
        'rnd_transpose': rnd_transpose,
        'rnd_colour': rnd_colour,
    }

    return model_opt


def get_ts_loggers(model_opt, restore_step=0):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['segm_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'segm_loss.csv'), ['train', 'valid'],
        name='Segmentation Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['iou'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'iou.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='IoU',
        buffer_size=1,
        restore_step=restore_step)
    loggers['wt_cov'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'wt_cov.csv'), ['train', 'valid'],
        name='Weighted Coverage',
        buffer_size=1,
        restore_step=restore_step)
    loggers['unwt_cov'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'unwt_cov.csv'), ['train', 'valid'],
        name='Unweighted Coverage',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dice'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'dice.csv'), ['train', 'valid'],
        name='Dice',
        buffer_size=1,
        restore_step=restore_step)
    loggers['learn_rate'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'learn_rate.csv'),
        'learning rate',
        name='Learning rate',
        buffer_size=1,
        restore_step=restore_step)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
        name='Step time',
        buffer_size=1,
        restore_step=restore_step)

    return loggers


def get_plot_loggers(model_opt, train_opt):
    samples = {}
    _ssets = ['train']
    if train_opt['has_valid']:
        _ssets.append('valid')
    for _set in _ssets:
        labels = ['input', 'output', 'total']
        for name in labels:
            key = '{}_{}'.format(name, _set)
            samples[key] = LazyRegisterer(
                os.path.join(logs_folder, '{}.png'.format(key)),
                'image', 'Samples {} {}'.format(name, _set))
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train a patch net')
    trainer.add_train_args(parser)
    trainer.add_data_args(parser)
    add_model_args(parser)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    tf.set_random_seed(1234)
    saver = None
    train_opt = trainer.make_train_opt(args)
    data_opt = trainer.make_data_opt(args)
    model_opt = make_model_opt(args)

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
        if train_opt['model_id']:
            model_id = train_opt['model_id']
        else:
            model_id = trainer.get_model_id('ris_patch')
        step = 0
        exp_folder = os.path.join(train_opt['results'], model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    if not train_opt['save_ckpt']:
        log.warning(
            'Checkpoints saving is turned off. Use --save_ckpt flag to save.')

    # Logger
    if train_opt['logs']:
        logs_folder = train_opt['logs']
        logs_folder = os.path.join(logs_folder, model_id)
        log = logger.get(os.path.join(logs_folder, 'raw.log'))
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
    m = patch_model.get_model(model_opt, device)

    log.info('Loading dataset')
    dataset = trainer.get_dataset(args.dataset, data_opt)
    if model_opt['fixed_order']:
        dataset['train']['label_segmentation'] = trainer.sort_by_segm_size(
            dataset['train']['label_segmentation'])
        dataset['valid']['label_segmentation'] = trainer.sort_by_segm_size(
            dataset['valid']['label_segmentation'])

    sess = tf.Session()

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series loggers
    loggers = {}
    if train_opt['logs']:
        log_manager = LogManager(logs_folder)
        loggers = get_ts_loggers(model_opt, restore_step=step)
        trainer.register_raw_logs(log_manager, log, model_opt, saver)
        samples = get_plot_loggers(model_opt, train_opt)
        log_url = 'http://{}/deep-dashboard?id={}'.format(
            train_opt['localhost'], model_id)
        log.info('Visualization can be viewed at: {}'.format(log_url))

    batch_size = args.batch_size
    log.info('Batch size: {}'.format(batch_size))
    num_ex_train = dataset['train']['input'].shape[0]
    get_batch_train = trainer.get_batch_fn(dataset['train'])
    log.info('Number of training examples: {}'.format(num_ex_train))

    if train_opt['has_valid']:
        num_ex_valid = dataset['valid']['input'].shape[0]
        get_batch_valid = trainer.get_batch_fn(dataset['valid'])
        log.info('Number of validation examples: {}'.format(num_ex_valid))

    def run_samples():
        """Samples"""
        def _run_samples(x, y, s, phase_train, fname_input, fname_output,
                         fname_total=None):

            _outputs = ['x_trans', 'y_gt_trans', 'y_out',
                        'match', 'attn_top_left', 'attn_bot_right']

            _max_items = trainer.get_max_items_per_row(x.shape[1], x.shape[2])

            order = get_permuted_order(s)
            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y, m['s_gt']: s, m['order']: order}
            _r = trainer.run_model(sess, m, _outputs, _feed_dict)

            trainer.plot_input(fname_input, x=_r['x_trans'],
                               y_gt=_r['y_gt_trans'],
                               s_gt=s, max_items_per_row=_max_items)

            trainer.plot_output(fname_output, y_out=_r['y_out'],
                                s_out=s,
                                match=_r['match'],
                                attn=(_r['attn_top_left'],
                                      _r['attn_bot_right']),
                                max_items_per_row=_max_items)

            if fname_total:
                trainer.plot_total_instances(fname_total, y_out=_r['y_out'],
                                             s_out=s,
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

            names = ['input', 'output', 'total']

            _run_samples(
                _x, _y, _s, _is_train,
                fname_input=samples['input_{}'.format(_set)].get_fname(),
                fname_output=samples['output_{}'.format(_set)].get_fname(),
                fname_total=samples['total_{}'.format(_set)].get_fname())

            if not samples['output_{}'.format(_set)].is_registered():
                for _name in names:
                    samples['{}_{}'.format(_name, _set)].register()
        pass

    def get_outputs_valid():
        _outputs = ['loss', 'segm_loss', 'iou_soft', 'iou_hard', 'wt_cov_soft',
                    'wt_cov_hard', 'unwt_cov_soft', 'unwt_cov_hard', 'dice']

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'segm_loss', 'iou_soft', 'iou_hard', 'wt_cov_soft',
                    'wt_cov_hard', 'unwt_cov_soft', 'unwt_cov_hard', 'dice',
                    'learn_rate']

        return _outputs

    def run_stats(step, sess, model, num_batch, batch_iter, outputs, write_log, phase_train):
        """Validation"""
        bat_sz_total = 0
        r = {}

        for bb in xrange(num_batch):
            _x, _y, _s = batch_iter.next()
            _order = get_permuted_order(_s)
            _feed_dict = {model['x']: _x, model['phase_train']: phase_train,
                          model['y_gt']: _y, model['s_gt']: _s,
                          model['order']: _order}
            _r = trainer.run_model(sess, model, outputs, _feed_dict)
            bat_sz = _x.shape[0]
            bat_sz_total += bat_sz

            for key in _r.iterkeys():
                if key in r:
                    r[key] += _r[key] * bat_sz
                else:
                    r[key] = _r[key] * bat_sz

        for key in r.iterkeys():
            r[key] = r[key] / bat_sz_total

        log.info('{:d} loss {:.4f}'.format(step, r['loss']))
        write_log(step, r)

        pass

    def write_log_valid(loggers):

        def write(step, r):
            loggers['loss'].add(step, ['', r['loss']])
            loggers['segm_loss'].add(step, ['', r['segm_loss']])
            loggers['iou'].add(step, ['', r['iou_soft'], '', r['iou_hard']])
            loggers['wt_cov'].add(step, ['', r['wt_cov_hard']])
            loggers['unwt_cov'].add(step, ['', r['unwt_cov_hard']])
            loggers['dice'].add(step, ['', r['dice']])

            pass

        return write

    def write_log_trainval(loggers):

        def write(step, r):
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['segm_loss'].add(step, [r['segm_loss'], ''])

            loggers['iou'].add(step, [r['iou_soft'], '', r['iou_hard'], ''])
            loggers['wt_cov'].add(step, [r['wt_cov_hard'], ''])
            loggers['unwt_cov'].add(step, [r['unwt_cov_hard'], ''])
            loggers['dice'].add(step, [r['dice'], ''])
            loggers['learn_rate'].add(step, r['learn_rate'])

            pass

        return write

    def train_step(step, x, y, s, order):
        """Train step"""
        def check_nan(var):
            # Check NaN.
            if np.isnan(var):
                log.error('NaN occurred.')
                raise Exception('NaN')

        _outputs = ['loss', 'train_step']
        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                      m['s_gt']: s, m['order']: order}
        _start_time = time.time()
        r = trainer.run_model(sess, m, _outputs, _feed_dict)
        _step_time = (time.time() - _start_time) * 1000
        check_nan(r['loss'])

        # Print statistics.
        if step % train_opt['steps_per_log'] == 0:
            log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, r['loss'],
                                                          _step_time))
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['step_time'].add(step, _step_time)

        pass

    def get_permuted_order(s, rnd=None):
        order = np.zeros(s.shape)
        num_ex = s.shape[0]
        max_num_objects = s.shape[1]
        for ii in xrange(num_ex):
            num_object = s[ii].sum()
            order_ = np.arange(num_object)
            if rnd is not None:
                rnd.shuffle(order_)
            order[ii, : num_object] = order_
            order[ii, num_object:] = np.arange(num_object, max_num_objects)

        return order

    def train_loop(step=0):
        """Train loop"""
        random = np.random.RandomState(step)

        if train_opt['has_valid']:
            batch_iter_valid = BatchIterator(num_ex_valid,
                                             batch_size=batch_size,
                                             get_fn=get_batch_valid,
                                             cycle=True,
                                             progress_bar=False)
            outputs_valid = get_outputs_valid()
        num_batch_valid = trainer.get_num_batch_valid(args.dataset)
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

            order = get_permuted_order(_s, random)

            # Run validation stats
            if train_opt['has_valid']:
                if step % train_opt['steps_per_valid'] == 0:
                    log.info('Running validation')
                    run_stats(step, sess, m, num_batch_valid,
                              batch_iter_valid,
                              outputs_valid, write_log_valid(loggers), False)
                    pass

            # Train stats
            if step % train_opt['steps_per_trainval'] == 0:
                log.info('Running train validation')
                run_stats(step, sess, m, num_batch_valid,
                          batch_iter_trainval,
                          outputs_trainval, write_log_trainval(loggers), True)
                pass

            # Plot samples
            if step % train_opt['steps_per_plot'] == 0:
                run_samples()
                pass

            # Train step
            train_step(step, _x, _y, _s, order)

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

    pass
