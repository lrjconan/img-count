import cslab_environ

import fnmatch
import logger
import os
import yaml
import tensorflow as tf

log = logger.get()

kModelOptFilename = 'model_opt.yaml'
kDatasetOptFilename = 'dataset_opt.yaml'


def save_ckpt(folder, sess, model_opt=None, data_opt=None, global_step=None):
    """Save checkpoint.

    Args:
        folder:
        sess:
        global_step:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    ckpt_path = os.path.join(folder, 'model.ckpt')
    log.info('Saving checkpoint to {}'.format(ckpt_path))
    tf_saver = tf.train.Saver(tf.all_variables())
    tf_saver.save(sess, ckpt_path, global_step=global_step)

    if model_opt is not None:
        with open(os.path.join(folder, kModelOptFilename), 'w') as f:
            yaml.dump(model_opt, f, default_flow_style=False)

    if data_opt is not None:
        with open(os.path.join(folder, kDatasetOptFilename), 'w') as f:
            yaml.dump(data_opt, f, default_flow_style=False)

    pass


def get_latest_ckpt(folder):
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

    latest_ckpt = os.path.join(folder, 'model.ckpt-{}'.format(latest_step))
    return (latest_ckpt, latest_step)


def get_ckpt_info(folder):
    """Get info of the latest checkpoint."""

    if not os.path.exists(folder):
        raise Exception('Folder "{}" does not exist'.format(folder))

    model_id = os.path.basename(folder.rstrip('/'))
    log.info('Restoring from {}'.format(folder))
    model_opt_fname = os.path.join(folder, kModelOptFilename)
    data_opt_fname = os.path.join(folder, kDatasetOptFilename)

    if os.path.exists(model_opt_fname):
        with open(model_opt_fname) as f_opt:
            model_opt = yaml.load(f_opt)
    else:
        model_opt = None

    log.info('Model options: {}'.format(model_opt))

    if os.path.exists(data_opt_fname):
        with open(data_opt_fname) as f_opt:
            data_opt = yaml.load(f_opt)
    else:
        data_opt = None

    ckpt_fname, latest_step = get_latest_ckpt(folder)
    log.info('Restoring at step {}'.format(latest_step))

    return {
        'ckpt_fname': ckpt_fname,
        'model_opt': model_opt,
        'data_opt': data_opt,
        'step': latest_step,
        'model_id': model_id
    }

def restore_ckpt(sess, ckpt_fname):
    """Restore the checkpoint file."""
    tf_saver = tf.train.Saver(tf.all_variables())
    tf_saver.restore(sess, ckpt_fname)

    pass
