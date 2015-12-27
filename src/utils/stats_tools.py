import logger
import numpy as np
import sys

log = logger.get()


def confusion_matrix(pred, labels):
    """Compute confusion matrix.

    Args:
        pred: list or numpy.ndarray, (N, 1), dtype: int, prediction vector.
        labels: list or numpy.ndarray, (N, 1), dtype: int, labels vector.
    Returns:
        cf_mat: numpy.ndarray, (K, K), dtype: int64, K is the number of 
        classes, unnormalized confusion matrix.
    """
    label_min = min(np.min(pred), np.min(labels))
    label_max = max(np.max(pred), np.max(labels))
    log.info('Label min: {}'.format(label_min))
    log.info('Label max: {}'.format(label_max))
    label_range = label_max - label_min + 1
    log.info('Label range: {}'.format(label_range))
    cf_mat = np.zeros((label_range, label_range), dtype='int64')
    if isinstance(pred, list):
        if len(pred) != len(labels):
            log.error('Length of prediction and labels do not match.')
        num_ex = len(pred)
    elif isinstance(pred, np.ndarray):
        if pred.shape[0] != labels.shape[0]:
            log.error('Length of prediction and labels do not match.')
        num_ex = pred.shape[0]
    else:
        log.error('Unknown type: {}'.format(type(pred)))

    for i in xrange(num_ex):
        cf_mat[pred[i] - label_min, labels[i] - label_min] += 1

    return cf_mat


def confusion_matrix_norm(pred, labels):
    """Compute normalized confusion matrix.

    Args:
        pred: list or numpy.ndarray, (N, 1), dtype: int, prediction vector.
        labels: list or numpy.ndarray, (N, 1), dtype: int, labels vector.
    Returns:
        cf_mat: numpy.ndarray, (K, K), dtype: int64, K is the number of 
        classes, normalized by the number of labels.
    """
    cf_mat = confusion_matrix(pred, labels)
    # label_min = min(np.min(pred), np.min(labels))
    # label_max = max(np.max(pred), np.max(labels))
    # log.info('Label min: {}'.format(label_min))
    # log.info('Label max: {}'.format(label_max))
    # label_range = label_max - label_min + 1
    # log.info('Label range: {}'.format(label_range))
    # label_count = np.zeros((label_range, 1), dtype='int64')

    # for i in xrange(label_min, label_max):
    #     labels_np = np.array(labels, dtype='int64')
    #     label_count[i - label_min] = np.sum((labels_np == i).astype('int64'))
    
    # cf_mat_norm = cf_mat / label_count.astype('float64')
    cf_mat_sum = np.sum(cf_mat, axis=0)
    cf_mat_norm = cf_mat / cf_mat_sum.astype('float64')
    cf_mat_norm[:, (cf_mat_sum == 0)] = 0.0

    return cf_mat_norm


def print_confusion_matrix(cf_mat, label_classes=None):
    """Print confusion matrix.

    Args:
        cf_mat: numpy.ndarray, (K, K), dtype: int or float, confusion matrix.
        label_classes: list, (K), dtype: string, label annotation.
    """
    log.info('----Confusion matrix----')

    if label_classes is None:
        label_classes = range(cf_mat.shape[0])

    sys.stdout.write('   P/L |')
    for i in xrange(len(label_classes)):
        sys.stdout.write('{:6}'.format(label_classes[i]))
    sys.stdout.write('\n')
    sys.stdout.write('{:8}'.format(''))
    for i in xrange(len(label_classes)):
        sys.stdout.write('------')
    sys.stdout.write('\n')

    for i in xrange(len(label_classes)):
        sys.stdout.write('{:6} |'.format(label_classes[i]))
        for j in xrange(len(label_classes)):
            if cf_mat.dtype == np.float64 or cf_mat.dtype == np.float32:
                sys.stdout.write('{:6.2f}'.format(cf_mat[i][j]))
            else:
                sys.stdout.write('{:6d}'.format(cf_mat[i][j]))
        sys.stdout.write('\n')
    log.info('------------------------')

    pass

if __name__ == '__main__':
    p = np.ceil(np.random.rand(20000) * 10).astype('int64')
    l = np.ceil(np.random.rand(20000) * 10).astype('int64')
    cm = confusion_matrix(p, l)
    cmn = confusion_matrix_norm(p, l)
    print_confusion_matrix(cm, range(1, 11))
    print_confusion_matrix(cmn, range(1, 11))
