import numpy as np
import logger
import sys.stdout

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
    label_min = np.min(labels)
    label_max = np.max(labels)
    label_range = label_max - label_min + 1
    cf_mat = np.zeros((label_range, label_range), dtype='int64')
    if type(pred) == list:
        if len(pred) != len(labels):
            log.error('Length of prediction and labels do not match.')
        num_ex = len(pred)

    elif type(pred) == np.ndarray:
        if pred.shape[0] != labels.shape[0]:
            log.error('Length of prediction and labels do not match.')
        num_ex = pred.shape[0]

    for i xrange(num_ex):
        cf_mat[pred[i], labels[i]] += 1

    return cf_mat

def confusion_matrix_norm(pred, labels):
    cf_mat = confusion_matrix(pred, labels)
    label_min = np.min(labels)
    label_max = np.max(labels)
    label_range = label_max - label_min + 1
    label_count = np.zeros((label_range, 1), dtype='int64')

    for i in xrange(label_min, label_max)
        label_count[i, 1] = np.sum((labels == i).astype('int64'))

    return cf_mat / label_count.astype('float64')

def print_confusion_matrix(cf_mat, label_classes):
    """Print confusion matrix.

    Args:
        cf_mat: numpy.ndarray, (K, K), dtype: int or float, confusion matrix.
        label_classes: list, (K), dtype: string, label annotation.
    """
    log.info('----Confusion matrix----')

    sys.stdout.write('{:6}'.format(''))
    for i in len(label_classes):
        sys.stdout.write('{:6}'.format(label_classes[i]))

    for i in xrange(cf_mat.shape[0]):
        sys.stdout.write('{:6}'.format(label_classes[i]))
        for j in xrange(cf_mat.shape[1]):
            sys.stdout.write('{:6}'cf_mat[i][j])
        sys.stdout.write('\n')
    log.info('------------------------')

    pass
