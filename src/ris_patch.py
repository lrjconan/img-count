"""
Patch instance segmentation network.

Usage: python ris_patch.py --help
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


def _add_dataset_args(parser):
    # Default dataset options
    kDataset = 'synth_shape'
    kHeight = 224
    kWidth = 224

    # (Below are only valid options for synth_shape dataset)
    kRadiusLower = 15
    kRadiusUpper = 45
    kBorderThickness = 3
    kNumExamples = 1000
    kMaxNumObjects = 6
    kNumObjectTypes = 1
    kSizeVar = 20
    kCenterVar = 20

    # Dataset options
    parser.add_argument('--dataset', default=kDataset)
    parser.add_argument('--height', default=kHeight, type=int)
    parser.add_argument('--width', default=kWidth, type=int)

    parser.add_argument('--radius_upper', default=kRadiusUpper, type=int)
    parser.add_argument('--radius_lower', default=kRadiusLower, type=int)
    parser.add_argument('--border_thickness', default=kBorderThickness)
    parser.add_argument('--num_ex', default=kNumExamples, type=int)
    parser.add_argument('--max_num_objects', default=kMaxNumObjects, type=int)
    parser.add_argument('--num_object_types',
                        default=kNumObjectTypes, type=int)
    parser.add_argument('--center_var', default=kCenterVar, type=float)
    parser.add_argument('--size_var', default=kSizeVar, type=float)

    pass


def _add_model_args(parser):
    kWeightDecay = 5e-5
    kBaseLearnRate = 1e-3
    kLearnRateDecay = 0.96
    kStepsPerLearnRateDecay = 5000
    kLossMixRatio = 1.0
    kMlpDropout = 0.5
    kPadding = 16

    kFilterHeight = 48
    kFilterWidth = 48
    kAttnBoxPaddingRatio = 0.2

    kAttnCnnFilterSize = '3,3,3'
    kAttnCnnDepth = '4,8,16'
    kAttnCnnPool = '2,2,2'
    kAttnDcnnFilterSize = '3,3,3,3'
    kAttnDcnnDepth = '16,8,4,1'
    kAttnDcnnPool = '2,2,2,1'

    kCtrlMlpDim = 256
    kNumCtrlMlpLayers = 2
    kCtrlRnnHiddenDim = 256
    kAttnRnnHiddenDim = 256
    kNumAttnMlpLayers = 2
    kAttnMlpDepth = 6
    kGtSelector = 'argmax'
    kKnobDecay = 0.9
    kStepsPerKnobDecay = 300
    kKnobBase = 1.0
    kKnobBoxOffset = 300
    kKnobSegmOffset = 500
    kGtBoxCtrNoise = 0.05
    kGtBoxPadNoise = 0.1
    kGtSegmNoise = 0.3
    kClipGradient = 1.0

    # Model type
    parser.add_argument('--model', default='attention',
                        help='Which model to train')

    # Shared options
    parser.add_argument('--weight_decay', default=kWeightDecay, type=float)
    parser.add_argument('--base_learn_rate', default=kBaseLearnRate)
    parser.add_argument('--learn_rate_decay', default=kLearnRateDecay)
    parser.add_argument('--steps_per_learn_rate_decay',
                        default=kStepsPerLearnRateDecay, type=int)
    parser.add_argument('--segm_loss_fn', default='iou')
    parser.add_argument('--mlp_dropout', default=kMlpDropout,
                        type=float)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--no_cum_min', action='store_true')
    parser.add_argument('--fixed_order', action='store_true')
    parser.add_argument('--padding', default=kPadding, type=int)

    # Attention-based model options
    parser.add_argument('--filter_height', default=kFilterHeight, type=int)
    parser.add_argument('--filter_width', default=kFilterWidth, type=int)
    parser.add_argument('--attn_cnn_filter_size', default=kAttnCnnFilterSize)
    parser.add_argument('--attn_cnn_depth', default=kAttnCnnDepth)
    parser.add_argument('--attn_cnn_pool', default=kAttnCnnPool)
    parser.add_argument('--attn_dcnn_filter_size', default=kAttnDcnnFilterSize)
    parser.add_argument('--attn_dcnn_depth', default=kAttnDcnnDepth)
    parser.add_argument('--attn_dcnn_pool', default=kAttnDcnnPool)
    parser.add_argument('--num_attn_mlp_layers',
                        default=kNumAttnMlpLayers, type=int)
    parser.add_argument('--attn_mlp_depth', default=kAttnMlpDepth, type=int)
    parser.add_argument('--attn_box_padding_ratio',
                        default=kAttnBoxPaddingRatio, type=float)
    parser.add_argument('--gt_box_ctr_noise',
                        default=kGtBoxCtrNoise, type=float)
    parser.add_argument('--gt_box_pad_noise',
                        default=kGtBoxPadNoise, type=float)
    parser.add_argument('--gt_segm_noise', default=kGtSegmNoise, type=float)
    parser.add_argument('--clip_gradient', default=kClipGradient, type=float)
    parser.add_argument('--add_skip_conn', action='store_true')
    pass


def _add_training_args(parser):
    # Training options
    parser.add_argument('--model_id', default=None)
    parser.add_argument('--num_steps', default=500000, type=int)
    parser.add_argument('--steps_per_ckpt', default=1000, type=int)
    parser.add_argument('--steps_per_valid', default=250, type=int)
    parser.add_argument('--steps_per_trainval', default=100, type=int)
    parser.add_argument('--steps_per_plot', default=50, type=int)
    parser.add_argument('--steps_per_log', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--results', default='../results')
    parser.add_argument('--logs', default='../results')
    parser.add_argument('--localhost', default='localhost')
    parser.add_argument('--restore', default=None)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--num_samples_plot', default=10, type=int)
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--debug_bn', action='store_true')
    parser.add_argument('--debug_act', action='store_true')
    parser.add_argument('--no_valid', action='store_true')
    parser.add_argument('--debug_weights', action='store_true')

    pass


def _make_model_opt(args):
    """Convert command-line arguments into model opt dict."""
    inp_height, inp_width, timespan = trainer.get_inp_dim(args.dataset)
    rnd_hflip, rnd_vflip, rnd_transpose, rnd_colour = \
        trainer.get_inp_transform(args.dataset)

    if args.dataset == 'synth_shape':
        timespan = args.max_num_objects + 1

    if args.model == 'attention':
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
            'use_bn': args.use_bn,
            'attn_box_padding_ratio': args.attn_box_padding_ratio,
            'gt_box_ctr_noise': args.gt_box_ctr_noise,
            'gt_box_pad_noise': args.gt_box_pad_noise,
            'gt_segm_noise': args.gt_segm_noise,
            'clip_gradient': args.clip_gradient,
            'fixed_order': args.fixed_order,
            'add_skip_conn': args.add_skip_conn,

            'rnd_hflip': rnd_hflip,
            'rnd_vflip': rnd_vflip,
            'rnd_transpose': rnd_transpose,
            'rnd_colour': rnd_colour,
        }

    return model_opt


def _make_data_opt(args):
    """Make command-line arguments into data opt dict."""
    inp_height, inp_width, timespan = trainer.get_inp_dim(args.dataset)
    if args.dataset == 'synth_shape':
        timespan = args.max_num_objects + 1

    if args.dataset == 'synth_shape':
        data_opt = {
            'height': inp_height,
            'width': inp_width,
            'timespan': timespan,
            'radius_upper': args.radius_upper,
            'radius_lower': args.radius_lower,
            'border_thickness': args.border_thickness,
            'max_num_objects': args.max_num_objects,
            'num_object_types': args.num_object_types,
            'center_var': args.center_var,
            'size_var': args.size_var,
            'num_train': args.num_ex,
            'num_valid': int(args.num_ex / 10),
            'has_valid': True
        }
    elif args.dataset == 'cvppp':
        data_opt = {
            'height': inp_height,
            'width': inp_width,
            'timespan': timespan,
            'num_train': None,
            'num_valid': None,
            'has_valid': not args.no_valid
        }
    elif args.dataset == 'kitti':
        data_opt = {
            'height': inp_height,
            'width': inp_width,
            'timespan': timespan,
            'num_train': args.num_ex,
            'num_valid': args.num_ex,
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
        'debug_bn': args.debug_bn,
        'debug_act': args.debug_act,
        'has_valid': not args.no_valid,
        'results': args.results,
        'restore': args.restore,
        'save_ckpt': args.save_ckpt,
        'logs': args.logs,
        'gpu': args.gpu,
        'localhost': args.localhost,
        'debug_weights': args.debug_weights
    }

    return train_opt


def _parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(
        description='Train an instance segmentation patch network')

    _add_dataset_args(parser)
    _add_model_args(parser)
    _add_training_args(parser)

    args = parser.parse_args()

    return args


def _get_ts_loggers(model_opt, debug_bn=False, debug_weights=False):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1)
    loggers['segm_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'segm_loss.csv'), ['train', 'valid'],
        name='Segmentation Loss',
        buffer_size=1)
    loggers['iou'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'iou.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='IoU',
        buffer_size=1)
    loggers['wt_cov'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'wt_cov.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='Weighted Coverage',
        buffer_size=1)
    loggers['unwt_cov'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'unwt_cov.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='Unweighted Coverage',
        buffer_size=1)
    loggers['learn_rate'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'learn_rate.csv'),
        'learning rate',
        name='Learning rate',
        buffer_size=1)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
        name='Step time',
        buffer_size=1)

    return loggers


def _get_plot_loggers(model_opt, train_opt):
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


if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)
    saver = None
    train_opt = _make_train_opt(args)
    model_opt = _make_model_opt(args)
    data_opt = _make_data_opt(args)
    # log.fatal(data_opt)

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
            'Checkpoints saving is turned off. Use -save_ckpt flag to save.')

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

    # gpu_options = tf.GPUOptions(per_procegss_gpu_memory_fraction=0.333)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series loggers
    loggers = {}
    if train_opt['logs']:
        log_manager = LogManager(logs_folder)
        loggers = _get_ts_loggers(model_opt, debug_bn=train_opt['debug_bn'],
                                  debug_weights=train_opt['debug_weights'])
        trainer.register_raw_logs(log_manager, log, model_opt, saver)
        samples = _get_plot_loggers(model_opt, train_opt)
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
        attn = model_opt['type'] == 'attention' or model_opt[
            'type'] == 'double_attention'

        def _run_samples(x, y, s, phase_train, fname_input, fname_output,
                         fname_total=None):

            _outputs = ['x_trans', 'y_gt_trans', 'y_out', 'match']

            if attn:
                _outputs.extend(['attn_top_left', 'attn_bot_right'])

            _max_items = trainer.get_max_items_per_row(x.shape[1], x.shape[2])

            order = get_permuted_order(s)
            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y, m['s_gt']: s, m['order']: order}
            _r = trainer.run_model(sess, m, _outputs, _feed_dict)

            trainer.plot_input(fname_input, x=_r['x_trans'],
                               y_gt=_r['y_gt_trans'],
                               s_gt=s, max_items_per_row=_max_items)

            if attn:
                trainer.plot_output(fname_output, y_out=_r['y_out'], s_out=s,
                                    match=_r['match'],
                                    attn=(_r['attn_top_left'],
                                          _r['attn_bot_right']),
                                    max_items_per_row=_max_items)
            else:
                trainer.plot_output(fname_output, y_out=_r['y_out'], s_out=s,
                                    match=_r['match'],
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
        _outputs = ['loss', 'segm_loss', 'iou_soft', 'iou_hard',
                    'wt_cov_soft', 'wt_cov_hard', 'unwt_cov_soft',
                    'unwt_cov_hard']

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'segm_loss', 'iou_soft', 'iou_hard', 'wt_cov_soft',
                    'wt_cov_hard', 'unwt_cov_soft', 'unwt_cov_hard',
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
            loggers['wt_cov'].add(step, ['', r['wt_cov_soft'], '',
                                         r['wt_cov_hard']])
            loggers['unwt_cov'].add(step, ['', r['unwt_cov_soft'], '',
                                           r['unwt_cov_hard']])

            pass

        return write

    def write_log_trainval(loggers):

        def write(step, r):
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['segm_loss'].add(step, [r['segm_loss'], ''])

            loggers['iou'].add(step, [r['iou_soft'], '', r['iou_hard'], ''])
            loggers['wt_cov'].add(step, [r['wt_cov_soft'], '',
                                         r['wt_cov_hard'], ''])
            loggers['unwt_cov'].add(step, [r['unwt_cov_soft'], '',
                                           r['unwt_cov_hard'], ''])
            loggers['learn_rate'].add(step, r['learn_rate'])

            pass

        return write

    def train_step(step, x, y, s, order):
        """Train step"""
        _outputs = ['loss', 'train_step']

        _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                      m['s_gt']: s, m['order']: order}
        _start_time = time.time()
        r = trainer.run_model(sess, m, _outputs, _feed_dict)
        _step_time = (time.time() - _start_time) * 1000

        # Print statistics.
        if step % train_opt['steps_per_log'] == 0:
            log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, r['loss'],
                                                          _step_time))
            loggers['loss'].add(step, [r['loss'], ''])
            loggers['step_time'].add(step, _step_time)

        # Check NaN.
        if np.isnan(r['loss']):
            saver.save(sess, global_step=step)
            input_file = h5py.File(
                os.path.join(exp_folder, 'nan_input.h5'), 'w')
            input_file['x'] = x
            input_file['y'] = y
            input_file['s'] = s
            input_file['order'] = order
            raise Exception('NaN occurred. Saved last step.')

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
        if train_opt['debug_bn']:
            if train_opt['has_valid']:
                outputs_valid.extend(get_outputs_bn())
            outputs_trainval.extend(get_outputs_bn())

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
