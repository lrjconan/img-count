"""
Train a box net. Usage: python ris_box.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import h5py
import numpy as np
import os
import tensorflow as tf
import time

from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.log_manager import LogManager
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger

import ris_box_model as box_model
import ris_train_base as trainer

log = logger.get()


def add_model_args(parser):
    parser.add_argument('--padding', default=16, type=int)
    parser.add_argument('--filter_height', default=48, type=int)
    parser.add_argument('--filter_width', default=48, type=int)
    parser.add_argument('--ctrl_cnn_filter_size', default='3,3,3,3,3,3,3,3')
    parser.add_argument('--ctrl_cnn_depth', default='4,4,8,8,16,16,32,64')
    parser.add_argument('--ctrl_cnn_pool', default='1,2,1,2,1,2,2,2')
    parser.add_argument('--box_loss_fn', default='iou')
    parser.add_argument('--fixed_order', action='store_true')
    parser.add_argument('--pretrain_cnn', default=None)
    parser.add_argument('--ctrl_rnn_hid_dim', default=256, type=int)
    parser.add_argument('--num_ctrl_mlp_layers', default=2, type=int)
    parser.add_argument('--ctrl_mlp_dim', default=256, type=int)
    parser.add_argument('--use_iou_box', action='store_true')
    parser.add_argument('--base_learn_rate', default=0.001, type=float)
    parser.add_argument('--learn_rate_decay', default=0.96, type=float)
    parser.add_argument('--steps_per_learn_rate_decay', default=5000, type=int)
    parser.add_argument('--squash_ctrl_params', action='store_true')
    parser.add_argument('--ctrl_rnn_inp_struct', default='dense')
    parser.add_argument('--num_ctrl_rnn_iter',  default=5, type=int)
    parser.add_argument('--num_glimpse_mlp_layers', default=2, type=int)

    pass


def make_model_opt(args):
    ccnn_fsize_list = args.ctrl_cnn_filter_size.split(',')
    ccnn_fsize_list = [int(fsize) for fsize in ccnn_fsize_list]
    ccnn_depth_list = args.ctrl_cnn_depth.split(',')
    ccnn_depth_list = [int(depth) for depth in ccnn_depth_list]
    ccnn_pool_list = args.ctrl_cnn_pool.split(',')
    ccnn_pool_list = [int(pool) for pool in ccnn_pool_list]

    inp_height, inp_width, timespan = trainer.get_inp_dim(args.dataset)

    model_opt = {
        'timespan': timespan,
        'inp_height': inp_height,
        'inp_width': inp_width,
        'inp_depth': 3,
        'padding': args.padding,
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
        'base_learn_rate': args.base_learn_rate,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
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


def get_ts_loggers(model_opt, restore_step=0):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['box_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'box_loss.csv'), ['train', 'valid'],
        name='Box Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['conf_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'conf_loss.csv'), ['train', 'valid'],
        name='Confidence Loss',
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
    _ssets = ['train', 'valid']
    for _set in _ssets:
        labels = ['input', 'output']
        if model_opt['ctrl_rnn_inp_struct'] == 'attn':
            labels.append('attn')
        for name in labels:
            key = '{}_{}'.format(name, _set)
            samples[key] = LazyRegisterer(
                os.path.join(logs_folder, '{}.png'.format(key)),
                'image', 'Samples {} {}'.format(name, _set))

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train a box net')
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
    model_opt_read = make_model_opt(args)
    data_opt = trainer.make_data_opt(args)

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
        model_opt['pretrain_cnn'] = None
    else:
        if train_opt['model_id']:
            model_id = train_opt['model_id']
        else:
            model_id = trainer.get_model_id('ris_box')
        model_opt = model_opt_read
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
    m = box_model.get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = trainer.get_dataset(args.dataset, data_opt)
    if model_opt['fixed_order']:
        dataset['train']['label_segmentation'] = trainer.sort_by_segm_size(
            dataset['train']['label_segmentation'])
        dataset['valid']['label_segmentation'] = trainer.sort_by_segm_size(
            dataset['valid']['label_segmentation'])

    sess = tf.Session()

    # Create time series loggers
    if train_opt['logs']:
        log_manager = LogManager(logs_folder)
        loggers = get_ts_loggers(model_opt, restore_step=step)
        trainer.register_raw_logs(log_manager, log, model_opt, saver)
        samples = get_plot_loggers(model_opt, train_opt)
        log_url = 'http://{}/deep-dashboard?id={}'.format(
            train_opt['localhost'], model_id)
        log.info('Visualization can be viewed at: {}'.format(log_url))

    # Restore/intialize weights
    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

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
        def _run_samples(x, y, s, phase_train, fname_input, fname_output, fname_attn=None):
            _outputs = ['x_trans', 'y_gt_trans', 'attn_top_left',
                        'attn_bot_right', 'attn_top_left_gt',
                        'attn_bot_right_gt', 'match_box', 's_out']

            if fname_attn:
                _outputs.append('ctrl_rnn_glimpse_map')
            _max_items = trainer.get_max_items_per_row(x.shape[1], x.shape[2])

            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y, m['s_gt']: s}
            _r = trainer.run_model(sess, m, _outputs, _feed_dict)

            _x_tile = np.expand_dims(_r['x_trans'], 1)
            _x_tile = np.tile(_x_tile, [1, y.shape[1], 1, 1, 1])

            trainer.plot_output(fname_input, y_out=_x_tile,
                                s_out=s,
                                match=_r['match_box'],
                                attn=(_r['attn_top_left_gt'],
                                      _r['attn_bot_right_gt']),
                                max_items_per_row=_max_items)

            trainer.plot_output(fname_output, y_out=_x_tile,
                                s_out=_r['s_out'],
                                match=_r['match_box'],
                                attn=(_r['attn_top_left'],
                                      _r['attn_bot_right']),
                                max_items_per_row=_max_items)

            if fname_attn:
                trainer.plot_double_attention(fname_attn, _r['x_trans'],
                                              _r['ctrl_rnn_glimpse_map'],
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

            labels = ['input', 'output']
            fname_input = samples['input_{}'.format(_set)].get_fname()
            fname_output = samples['output_{}'.format(_set)].get_fname()
            if model_opt['ctrl_rnn_inp_struct'] == 'attn':
                fname_attn = samples['attn_{}'.format(_set)].get_fname()
                labels.append('attn')
            else:
                fname_attn = None

            _run_samples(_x, _y, _s, _is_train,
                         fname_input=fname_input,
                         fname_output=fname_output,
                         fname_attn=fname_attn)

            if not samples['output_{}'.format(_set)].is_registered():
                for _name in labels:
                    samples['{}_{}'.format(_name, _set)].register()

        pass

    def get_outputs_valid():
        _outputs = ['loss', 'box_loss', 'conf_loss']

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'box_loss', 'conf_loss']

        return _outputs

    def write_log_valid(loggers):

        def write(step, r):
            loggers['loss'].add(step, ['', r['loss']])
            loggers['box_loss'].add(step, ['', r['box_loss']])
            loggers['conf_loss'].add(step, ['', r['conf_loss']])

            pass

        return write

    def write_log_trainval(loggers):

        def write(step, r):
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['box_loss'].add(step, [r['box_loss'], ''])
            loggers['conf_loss'].add(step, [r['conf_loss'], ''])

            pass

        return write

    def train_step(step, x, y, s):
        """Train step"""
        def check_nan(var):
            # Check NaN.
            if np.isnan(var):
                log.error('NaN occurred.')
                raise Exception('NaN')

        _outputs = ['loss', 'train_step']
        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                      m['s_gt']: s}
        _start_time = time.time()
        r = trainer.run_model(sess, m, _outputs, _feed_dict)
        _step_time = (time.time() - _start_time) * 1000
        check_nan(r['loss'])

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
            # Run validation stats
            if train_opt['has_valid']:
                if step % train_opt['steps_per_valid'] == 0:
                    log.info('Running validation')
                    trainer.run_stats(step, sess, m, num_batch_valid,
                                      batch_iter_valid,
                                      outputs_valid, write_log_valid(loggers),
                                      False)
                    pass

            # Train stats
            if step % train_opt['steps_per_trainval'] == 0:
                log.info('Running train validation')
                trainer.run_stats(step, sess, m, num_batch_valid,
                                  batch_iter_trainval, outputs_trainval,
                                  write_log_trainval(loggers), True)
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
