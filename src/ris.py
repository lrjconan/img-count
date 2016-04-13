"""
Train a ris net. Usage: python ris.py --help
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
from utils import plot_utils as pu
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.log_manager import LogManager
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger

import ris_vanilla_model as vanilla_model
import ris_attn_model as attention_model
import ris_train_base as trainer

log = logger.get()


def get_model(opt, device='/cpu:0'):
    """Model router."""
    name = opt['type']
    if name == 'vanilla':
        return vanilla_model.get_model(opt, device=device)
    elif name == 'attention':
        return attention_model.get_model(opt, device=device)
    else:
        raise Exception('Unknown model name "{}"'.format(name))

    pass


def add_model_args(parser):
    # Model type
    parser.add_argument('--model', default='attention')

    # Original model options
    parser.add_argument('--cnn_filter_size', default='3,3,3,3,3')
    parser.add_argument('--cnn_depth', default='4,8,8,12,16')
    parser.add_argument('--cnn_pool', default='2,2,2,2,2')
    parser.add_argument('--dcnn_filter_size', default='3,3,3,3,3,3')
    parser.add_argument('--dcnn_depth', default='8,6,4,4,2,1')
    parser.add_argument('--dcnn_pool', default='2,2,2,2,2,1')
    parser.add_argument('--rnn_type', default='lstm')
    parser.add_argument('--conv_lstm_filter_size', default=3)
    parser.add_argument('--conv_lstm_hid_depth', default=12)
    parser.add_argument('--rnn_hid_dim', default=256)
    parser.add_argument('--score_maxpool', default=1, type=int)
    parser.add_argument('--num_mlp_layers', default=2, type=int)
    parser.add_argument('--mlp_depth', default=6, type=int)
    parser.add_argument('--use_deconv', action='store_true')
    parser.add_argument('--score_use_core', action='store_true')

    # Shared options
    parser.add_argument('--padding', default=16, type=int)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--base_learn_rate', default=0.001)
    parser.add_argument('--learn_rate_decay', default=0.96, type=float)
    parser.add_argument('--steps_per_learn_rate_decay', default=5000, type=int)
    parser.add_argument('--loss_mix_ratio', default=1.0, type=float)
    parser.add_argument('--segm_loss_fn', default='iou')
    parser.add_argument('--mlp_dropout', default=None, type=float)
    parser.add_argument('--fixed_order', action='store_true')

    # Attention-based model options
    parser.add_argument('--filter_height', default=48, type=int)
    parser.add_argument('--filter_width', default=48, type=int)
    parser.add_argument('--ctrl_cnn_filter_size', default='3,3,3,3,3')
    parser.add_argument('--ctrl_cnn_depth', default='4,8,16,16,32')
    parser.add_argument('--ctrl_cnn_pool', default='2,2,2,2,2')
    parser.add_argument('--attn_cnn_filter_size', default='3,3,3')
    parser.add_argument('--attn_cnn_depth', default='4,8,16')
    parser.add_argument('--attn_cnn_pool', default='2,2,2')
    parser.add_argument('--attn_dcnn_filter_size', default='3,3,3,3')
    parser.add_argument('--attn_dcnn_depth', default='16,8,4,1')
    parser.add_argument('--attn_dcnn_pool', default='2,2,2,1')
    parser.add_argument('--ctrl_rnn_hid_dim', default=256, type=int)
    parser.add_argument('--num_ctrl_mlp_layers', default=1, type=int)
    parser.add_argument('--ctrl_mlp_dim', default=256, type=int)
    parser.add_argument('--num_attn_mlp_layers', default=1, type=int)
    parser.add_argument('--attn_mlp_depth', default=6, type=int)
    parser.add_argument('--box_loss_fn', default='iou')
    parser.add_argument('--attn_box_padding_ratio', default=0.2, type=float)
    parser.add_argument('--use_knob', action='store_true')
    parser.add_argument('--knob_decay', default=0.9, type=float)
    parser.add_argument('--steps_per_knob_decay', default=300)
    parser.add_argument('--knob_base', default=1.0, type=float)
    parser.add_argument('--knob_box_offset', default=300, type=int)
    parser.add_argument('--knob_segm_offset', default=500, type=int)
    parser.add_argument('--knob_use_timescale', action='store_true')
    parser.add_argument('--gt_box_ctr_noise', default=0.05, type=float)
    parser.add_argument('--gt_box_pad_noise', default=0.1, type=float)
    parser.add_argument('--gt_segm_noise', default=0.3)
    parser.add_argument('--clip_gradient', default=1.0, type=float)
    parser.add_argument('--squash_ctrl_params', action='store_true')
    parser.add_argument('--fixed_gamma', action='store_true')
    parser.add_argument('--pretrain_ctrl_net', default=None)
    parser.add_argument('--pretrain_attn_net', default=None)
    parser.add_argument('--pretrain_net', default=None)
    parser.add_argument('--freeze_ctrl_cnn', action='store_true')
    parser.add_argument('--freeze_ctrl_rnn', action='store_true')
    parser.add_argument('--freeze_attn_net', action='store_true')
    parser.add_argument('--ctrl_rnn_inp_struct', default='dense')
    parser.add_argument('--num_ctrl_rnn_iter', default=5, type=int)
    parser.add_argument('--num_glimpse_mlp_layers', default=2, type=int)

    pass


def make_model_opt(args):
    """Convert command-line arguments into model opt dict."""
    inp_height, inp_width, timespan = trainer.get_inp_dim(args.dataset)
    rnd_hflip, rnd_vflip, rnd_transpose, rnd_colour = \
        trainer.get_inp_transform(args.dataset)

    if args.dataset == 'synth_shape':
        timespan = args.max_num_objects + 1

    if args.model == 'attention':
        ccnn_fsize_list = args.ctrl_cnn_filter_size.split(',')
        ccnn_fsize_list = [int(fsize) for fsize in ccnn_fsize_list]
        ccnn_depth_list = args.ctrl_cnn_depth.split(',')
        ccnn_depth_list = [int(depth) for depth in ccnn_depth_list]
        ccnn_pool_list = args.ctrl_cnn_pool.split(',')
        ccnn_pool_list = [int(pool) for pool in ccnn_pool_list]

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
            'type': args.model,
            'inp_height': inp_height,
            'inp_width': inp_width,
            'inp_depth': 3,
            'padding': args.padding,
            'filter_height': args.filter_height,
            'filter_width': args.filter_width,
            'timespan': timespan,

            'ctrl_cnn_filter_size': ccnn_fsize_list,
            'ctrl_cnn_depth': ccnn_depth_list,
            'ctrl_cnn_pool': ccnn_pool_list,

            'ctrl_rnn_hid_dim': args.ctrl_rnn_hid_dim,

            'attn_cnn_filter_size': acnn_fsize_list,
            'attn_cnn_depth': acnn_depth_list,
            'attn_cnn_pool': acnn_pool_list,

            'attn_dcnn_filter_size': attn_dcnn_fsize_list,
            'attn_dcnn_depth': attn_dcnn_depth_list,
            'attn_dcnn_pool': attn_dcnn_pool_list,

            'num_ctrl_mlp_layers': args.num_ctrl_mlp_layers,
            'ctrl_mlp_dim': args.ctrl_mlp_dim,

            'attn_mlp_depth': args.attn_mlp_depth,
            'num_attn_mlp_layers': args.num_attn_mlp_layers,
            'mlp_dropout': args.mlp_dropout,

            'weight_decay': args.weight_decay,
            'base_learn_rate': args.base_learn_rate,
            'learn_rate_decay': args.learn_rate_decay,
            'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
            'loss_mix_ratio': args.loss_mix_ratio,

            'segm_loss_fn': args.segm_loss_fn,
            'box_loss_fn': args.box_loss_fn,
            'use_bn': True,
            'attn_box_padding_ratio': args.attn_box_padding_ratio,
            'use_knob': args.use_knob,
            'knob_decay': args.knob_decay,
            'knob_base': args.knob_base,
            'steps_per_knob_decay': args.steps_per_knob_decay,
            'knob_box_offset': args.knob_box_offset,
            'knob_segm_offset': args.knob_segm_offset,
            'knob_use_timescale': args.knob_use_timescale,
            'gt_box_ctr_noise': args.gt_box_ctr_noise,
            'gt_box_pad_noise': args.gt_box_pad_noise,
            'gt_segm_noise': args.gt_segm_noise,
            'squash_ctrl_params': args.squash_ctrl_params,
            'clip_gradient': args.clip_gradient,
            'fixed_order': args.fixed_order,
            'fixed_gamma': args.fixed_gamma,

            'ctrl_rnn_inp_struct': args.ctrl_rnn_inp_struct,
            'num_ctrl_rnn_iter': args.num_ctrl_rnn_iter,
            'num_glimpse_mlp_layers': args.num_glimpse_mlp_layers,

            'pretrain_ctrl_net': args.pretrain_ctrl_net,
            'pretrain_attn_net': args.pretrain_attn_net,
            'pretrain_net': args.pretrain_net,
            'freeze_ctrl_cnn': args.freeze_ctrl_cnn,
            'freeze_ctrl_rnn': args.freeze_ctrl_rnn,
            'freeze_attn_net': args.freeze_attn_net,

            'rnd_hflip': rnd_hflip,
            'rnd_vflip': rnd_vflip,
            'rnd_transpose': rnd_transpose,
            'rnd_colour': rnd_colour,
        }
    elif args.model == 'vanilla':
        cnn_fsize_list = args.cnn_filter_size.split(',')
        cnn_fsize_list = [int(fsize) for fsize in cnn_fsize_list]
        cnn_depth_list = args.cnn_depth.split(',')
        cnn_depth_list = [int(depth) for depth in cnn_depth_list]
        cnn_pool_list = args.cnn_pool.split(',')
        cnn_pool_list = [int(pool) for pool in cnn_pool_list]

        dcnn_fsize_list = args.dcnn_filter_size.split(',')
        dcnn_fsize_list = [int(fsize) for fsize in dcnn_fsize_list]
        dcnn_depth_list = args.dcnn_depth.split(',')
        dcnn_depth_list = [int(depth) for depth in dcnn_depth_list]
        dcnn_pool_list = args.dcnn_pool.split(',')
        dcnn_pool_list = [int(pool) for pool in dcnn_pool_list]

        model_opt = {
            'type': args.model,
            'inp_height': args.height,
            'inp_width': args.width,
            'inp_depth': 3,
            'padding': args.padding,
            'timespan': timespan,
            'weight_decay': args.weight_decay,
            'base_learn_rate': args.base_learn_rate,
            'learn_rate_decay': args.learn_rate_decay,
            'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
            'loss_mix_ratio': args.loss_mix_ratio,
            'cnn_filter_size': cnn_fsize_list,
            'cnn_depth': cnn_depth_list,
            'dcnn_filter_size': dcnn_fsize_list,
            'dcnn_depth': dcnn_depth_list,
            'rnn_type': args.rnn_type,
            'conv_lstm_filter_size': args.conv_lstm_filter_size,
            'conv_lstm_hid_depth': args.conv_lstm_hid_depth,
            'rnn_hid_dim': args.rnn_hid_dim,
            'mlp_depth': args.mlp_depth,
            'score_maxpool': args.score_maxpool,
            'num_mlp_layers': args.num_mlp_layers,
            'mlp_dropout': args.mlp_dropout,
            'segm_loss_fn': args.segm_loss_fn,
            'use_deconv': True,
            'use_bn': True,
            'segm_dense_conn': True,
            'add_skip_conn': True,
            'score_use_core': True,
            'clip_gradient': args.clip_gradient,
            'fixed_order': args.fixed_order,

            'rnd_hflip': rnd_hflip,
            'rnd_vflip': rnd_vflip,
            'rnd_transpose': rnd_transpose,
            'rnd_colour': rnd_colour
        }

    return model_opt


def get_ts_loggers(model_opt, restore_step=0):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['conf_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'conf_loss.csv'), ['train', 'valid'],
        name='Confidence Loss',
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
    loggers['dic'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'dic.csv'), ['train', 'valid'],
        name='DiC',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dic_abs'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'dic_abs.csv'), ['train', 'valid'],
        name='|DiC|',
        buffer_size=1,
        restore_step=restore_step)
    loggers['learn_rate'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'learn_rate.csv'), 'learning rate',
        name='Learning rate',
        buffer_size=1,
        restore_step=restore_step)
    loggers['count_acc'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'count_acc.csv'), ['train', 'valid'],
        name='Count acc',
        buffer_size=1,
        restore_step=restore_step)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
        name='Step time',
        buffer_size=1,
        restore_step=restore_step)

    if model_opt['type'] == 'attention':
        loggers['box_loss'] = TimeSeriesLogger(
            os.path.join(logs_folder, 'box_loss.csv'), ['train', 'valid'],
            name='Box Loss',
            buffer_size=1,
            restore_step=restore_step)
        loggers['gt_knob'] = TimeSeriesLogger(
            os.path.join(logs_folder, 'gt_knob.csv'),
            ['box', 'segmentation'],
            name='GT mix',
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
        if model_opt['type'] == 'attention':
            num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
            num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
            num_attn_dcnn = len(model_opt['attn_dcnn_filter_size'])
            labels.extend(['box', 'patch'])
            if model_opt['ctrl_rnn_inp_struct'] == 'attn':
                labels.append('attn')

        for name in labels:
            key = '{}_{}'.format(name, _set)
            samples[key] = LazyRegisterer(
                os.path.join(logs_folder, '{}.png'.format(key)),
                'image', 'Samples {} {}'.format(name, _set))

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train a ris net')
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
    model_opt_read = make_model_opt(args)

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
        model_opt['pretrain_attn_net'] = None
        model_opt['pretrain_ctrl_net'] = None
        model_opt['freeze_attn_net'] = model_opt_read['freeze_attn_net']
        model_opt['freeze_ctrl_cnn'] = model_opt_read['freeze_ctrl_cnn']
        model_opt['freeze_ctrl_rnn'] = model_opt_read['freeze_ctrl_rnn']
    else:
        if train_opt['model_id']:
            model_id = train_opt['model_id']
        else:
            model_id = trainer.get_model_id('ris')
        model_opt = model_opt_read
        step = 0
        exp_folder = os.path.join(train_opt['results'], model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    if not train_opt['save_ckpt']:
        log.warning(
            'Checkpoints saving is turned off. Use --save_ckpt flag to save.')

    if model_opt['type'] == 'attention':
        num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
        num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
        num_attn_dcnn = len(model_opt['attn_dcnn_filter_size'])

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
    m = get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = trainer.get_dataset(args.dataset, data_opt)

    if model_opt['fixed_order']:
        dataset['train']['label_segmentation'] = trainer.sort_by_segm_size(
            dataset['train']['label_segmentation'])
        dataset['valid']['label_segmentation'] = trainer.sort_by_segm_size(
            dataset['valid']['label_segmentation'])

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series loggers
    loggers = {}
    if train_opt['logs']:
        log_manager = LogManager(logs_folder)
        loggers = get_ts_loggers(model_opt)
        trainer.register_raw_logs(log_manager, log, model_opt, saver)
        samples = get_plot_loggers(model_opt, train_opt)
        _log_url = 'http://{}/deep-dashboard?id={}'.format(
            train_opt['localhost'], model_id)
        log.info('Visualization can be viewed at: {}'.format(_log_url))

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
        attn = model_opt['type'] == 'attention'

        def _run_samples(x, y, s, phase_train, fname_input, fname_output,
                         fname_total=None, fname_box=None, fname_patch=None,
                         fname_attn=None):

            _outputs = ['x_trans', 'y_gt_trans', 'y_out',
                        's_out', 'match']

            if attn:
                _outputs.extend(['attn_top_left', 'attn_bot_right',
                                 'attn_box', 'attn_box_gt', 'match_box'])
            if fname_attn:
                _outputs.append('ctrl_rnn_glimpse_map')

            if fname_patch:
                _outputs.append('x_patch')

            _max_items = trainer.get_max_items_per_row(x.shape[1], x.shape[2])


            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y, m['s_gt']: s}
            _r = trainer.run_model(sess, m, _outputs, _feed_dict)

            trainer.plot_input(fname_input, x=_r['x_trans'],
                               y_gt=_r['y_gt_trans'], s_gt=s,
                               max_items_per_row=_max_items)

            _x_tile = np.expand_dims(_r['x_trans'], 1)
            _x_tile = np.tile(_x_tile, [1, y.shape[1], 1, 1, 1])

            if attn:
                trainer.plot_output(fname_output, y_out=_r['y_out'],
                                    s_out=_r['s_out'],
                                    match=_r['match'],
                                    attn=(_r['attn_top_left'],
                                          _r['attn_bot_right']),
                                    max_items_per_row=_max_items)
            else:
                trainer.plot_output(fname_output, y_out=_r['y_out'],
                                    s_out=_r['s_out'],
                                    match=_r['match'],
                                    max_items_per_row=_max_items)

            if fname_total:
                trainer.plot_total_instances(fname_total, y_out=_r['y_out'],
                                             s_out=_r['s_out'],
                                             max_items_per_row=_max_items)

            if fname_box:
                trainer.plot_output(fname_box, y_out=_x_tile,
                                    s_out=_r['s_out'],
                                    match=_r['match_box'],
                                    attn=(_r['attn_top_left'],
                                          _r['attn_bot_right']),
                                    max_items_per_row=_max_items)

            if fname_patch:
                pu.plot_thumbnails(fname_patch,
                                   _r['x_patch'][:, :, :, :, : 3],
                                   axis=1, max_items_per_row=8)

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

            fname_input = samples['input_{}'.format(_set)].get_fname()
            fname_output = samples['output_{}'.format(_set)].get_fname()
            fname_total = samples['total_{}'.format(_set)].get_fname()

            if attn:
                fname_box = samples['box_{}'.format(_set)].get_fname()
                fname_patch = samples['patch_{}'.format(_set)].get_fname()
                names = ['input', 'output', 'total', 'box', 'patch']
            else:
                fname_box = None
                fname_patch = None
                names = ['input', 'output', 'total']

            if attn and model_opt['ctrl_rnn_inp_struct'] == 'attn':
                fname_attn = samples['attn_{}'.format(_set)].get_fname()
                names.append('attn')
            else:
                fname_attn = None

            _run_samples(_x, _y, _s, _is_train,
                         fname_input=fname_input,
                         fname_output=fname_output,
                         fname_total=fname_total,
                         fname_box=fname_box,
                         fname_patch=fname_patch,
                         fname_attn=fname_attn)

            if not samples['output_{}'.format(_set)].is_registered():
                for _name in names:
                    samples['{}_{}'.format(_name, _set)].register()

        pass

    def get_outputs_valid(model):
        _outputs = ['loss', 'conf_loss', 'segm_loss', 'iou_soft', 'iou_hard',
                    'count_acc', 'dice', 'dic', 'dic_abs', 'wt_cov_hard',
                    'unwt_cov_hard']

        if 'box_loss' in model:
            _outputs.extend(['box_loss'])

        return _outputs

    def get_outputs_trainval(model):
        _outputs = ['loss', 'conf_loss', 'segm_loss', 'iou_soft', 'iou_hard',
                    'count_acc', 'dice', 'dic', 'dic_abs', 'wt_cov_hard',
                    'unwt_cov_hard', 'learn_rate']

        if 'box_loss' in model:
            _outputs.extend(['box_loss', 'gt_knob_prob_box',
                             'gt_knob_prob_segm', 'attn_lg_gamma_mean',
                             'attn_box_lg_gamma_mean', 'y_out_lg_gamma_mean'])

        return _outputs

    def write_log_valid(loggers):

        def write(step, r):
            loggers['loss'].add(step, ['', r['loss']])
            loggers['conf_loss'].add(step, ['', r['conf_loss']])
            loggers['segm_loss'].add(step, ['', r['segm_loss']])
            if 'box_loss' in r:
                loggers['box_loss'].add(step, ['', r['box_loss']])
            loggers['iou'].add(step, ['', r['iou_soft'], '', r['iou_hard']])
            loggers['wt_cov'].add(step, ['', r['wt_cov_hard']])
            loggers['unwt_cov'].add(step, ['', r['unwt_cov_hard']])
            loggers['dice'].add(step, ['', r['dice']])
            loggers['count_acc'].add(step, ['', r['count_acc']])
            loggers['dic'].add(step, ['', r['dic']])
            loggers['dic_abs'].add(step, ['', r['dic_abs']])

            pass

        return write

    def write_log_trainval(loggers):

        def write(step, r):
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['conf_loss'].add(step, [r['conf_loss'], ''])
            loggers['segm_loss'].add(step, [r['segm_loss'], ''])
            if 'box_loss' in r:
                loggers['box_loss'].add(step, [r['box_loss'], ''])

            loggers['iou'].add(step, [r['iou_soft'], '', r['iou_hard'], ''])
            loggers['wt_cov'].add(step, [r['wt_cov_hard'], ''])
            loggers['unwt_cov'].add(step, [r['unwt_cov_hard'], ''])
            loggers['dice'].add(step, [r['dice'], ''])
            loggers['count_acc'].add(step, [r['count_acc'], ''])
            loggers['dic'].add(step, [r['dic'], ''])
            loggers['dic_abs'].add(step, [r['dic_abs'], ''])
            if 'gt_knob_prob_box' in r:
                loggers['gt_knob'].add(step, [r['gt_knob_prob_box'],
                                              r['gt_knob_prob_segm']])
            loggers['learn_rate'].add(step, r['learn_rate'])

            pass

        return write

    def train_step(step, x, y, s):
        """Train step"""
        def check_nan(var):
            # Check NaN.
            if np.isnan(var):
                log.error('NaN occurred.')
                raise Exception('NaN')

        _start_time = time.time()
        _outputs = ['loss', 'train_step']
        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                      m['s_gt']: s}
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

    def train_loop(step=0):
        """Train loop"""
        if train_opt['has_valid']:
            batch_iter_valid = BatchIterator(num_ex_valid,
                                             batch_size=batch_size,
                                             get_fn=get_batch_valid,
                                             cycle=True,
                                             progress_bar=False)
            outputs_valid = get_outputs_valid(m)
        num_batch_valid = trainer.get_num_batch_valid(args.dataset)
        batch_iter_trainval = BatchIterator(num_ex_train,
                                            batch_size=batch_size,
                                            get_fn=get_batch_train,
                                            cycle=True,
                                            progress_bar=False)
        outputs_trainval = get_outputs_trainval(m)

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

    pass
