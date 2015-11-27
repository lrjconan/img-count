"""
Generate counting questions and answers in MS-COCO dataset, using groundtruth 
annotation.
Answer distribution is reshaped into COCO-QA distribution

Usage: python mscoco_gen_count_data.py
"""

from data_api import MSCOCO
from data_api import COCOQA
from utils import logger
from utils import progress_bar
from utils.sharded_hdf5 import ShardedFile, ShardedFileWriter
import argparse
import numpy as np
import os

log = logger.get()

MAX_COUNT = 30


def get_count(annotations, cat_list_rev):
    """Get count information for each object class.

    Returns:
        count_dict: dict,
        {
            'image_id': string.
            'count': K-dim vector.
        }
    """
    num_classes = len(cat_list_rev)
    cat_counts = np.zeros((num_classes), dtype='int64')
    for ann in annotations:
        cat_id = cat_list_rev[ann['category_id']]
        cat_counts[cat_id] += 1

    counts_array = np.zeros((MAX_COUNT), dtype='int64')

    questions = []
    for i in xrange(num_classes):
        counts_array[cat_counts[i]] += 1
        if cat_counts[i] > 0:
            questions.append({'image_id': annotations[0]['image_id'],
                              'category': i,
                              'number': cat_counts[i]})

    count_dict = {
        'image_id': annotations[0]['image_id'],
        'cat_counts': cat_counts,
        'counts_array': counts_array,
        'questions': questions
    }

    return count_dict


def stat_count(counts_array):
    counts_sum = counts_array.sum(axis=0)
    counts_avg = counts_array.mean(axis=0)
    counts_dist = counts_sum / float(counts_sum.sum())
    log.info('Sum: {}'.format(counts_sum))
    log.info('Mean: {}'.format(counts_avg))
    log.info('Dist: {}'.format(counts_dist))

    pass


def apply_distribution(questions, dist):
    """Apply a distribution on data. Reject too common labels.
    """
    accum_count = np.zeros(dist.shape, dtype='float32')
    accum_dist = np.zeros(dist.shape, dtype='float32')
    keep = []
    random = np.random.RandomState(2)
    random.shuffle(questions)
    count = np.zeros(dist.shape, dtype='int64')
    for i, q in enumerate(questions):
        num = q['number']
        if num > 0 and num <= dist.size:
            count[num - 1] += 1
    orig_dist = count / count.sum().astype('float32')
    dist_ratio = orig_dist / dist
    least_idx = np.argmin(dist_ratio)
    total_admit = int(np.floor(count[least_idx] / dist[least_idx]))
    remainder = np.zeros(dist.shape, dtype='int64')
    for i in xrange(dist.size):
        remainder[i] = int(np.floor(dist[i] * total_admit))

    for i, q in enumerate(questions):
        num = q['number']
        if num > 0 and num <= dist.size:
            if remainder[num - 1] > 0:
                # if accum_dist[num - 1] < dist[num - 1] + 0.03 or \
                    # (accum_dist >= dist).all():
                # if random.uniform(0, 1, 1) <= dist[num - 1] / orig_dist[num -
                # 1]:
                keep.append(i)
                accum_count[num - 1] += 1
                remainder[num - 1] -= 1
                accum_dist = accum_count / accum_count.sum()

    log.info('Admitted {:d}/{:d}'.format(len(keep), len(questions)))
    log.info('Final count: {}'.format(accum_count))
    log.info('Final distribution: {}'.format(accum_dist))

    return keep


def stat_cocoqa(cocoqa):
    """Run statistics on COCO-QA"""
    qtypes = cocoqa.get_question_types()
    qtype_dict = cocoqa.get_question_type_dict()
    number_qids = []
    for i, qt in enumerate(qtypes):
        if qt == qtype_dict['number']:
            number_qids.append(i)

    ans_idict = cocoqa.get_answer_vocab_inv_dict()
    answers = cocoqa.get_encoded_answers()
    answers = answers[number_qids]
    dist = [0] * len(ans_idict)

    for a in answers:
        dist[a] += 1
    for i in xrange(len(dist)):
        if dist[i] != 0:
            log.info('{}: {}'.format(ans_idict[i], dist[i]))

    number_dict = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10
    }
    dist2 = [0] * 10

    for i in xrange(len(dist)):
        if dist[i] != 0:
            dist2[number_dict[ans_idict[i]] - 1] = dist[i]
    dist2 = np.array(dist2, dtype='float32')
    dist2 = dist2 / dist2.sum()

    return dist2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Generate counting questions')
    parser.add_argument('-set', default='train', help='Train or valid set')
    parser.add_argument('-mscoco_datadir', default='../data/mscoco',
                        help='Dataset directory')
    parser.add_argument('-cocoqa_datadir', default='../data/cocoqa',
                        help='Dataset directory')
    parser.add_argument('-output', default=None, help='Output file name')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    mscoco = MSCOCO(base_dir=args.mscoco_datadir, set_name=args.set)
    image_ids = mscoco.get_image_ids()
    cat_list_rev = mscoco.get_cat_list_reverse()

    counts_array = []
    cat_counts = []
    counts_id = []
    questions = []
    counts_dict_list = []
    err_image_ids = []
    empty_image_ids = []
    success_image_ids = []
    log.info('Scanning dataset')
    for image_id in progress_bar.get_iter(image_ids):
        annotations = mscoco.get_image_annotations(image_id)
        if annotations is not None and len(annotations) > 0:
            counts_dict = get_count(annotations, cat_list_rev)
            counts_array.append(counts_dict['counts_array'])
            cat_counts.append(counts_dict['cat_counts'])
            counts_id.append(counts_dict['image_id'])
            questions.extend(counts_dict['questions'])
            success_image_ids.append(image_id)
            counts_dict_list.append(counts_dict)
        elif annotations is None:
            err_image_ids.append(image_id)
        elif len(annotations) == 0:
            empty_image_ids.append(image_id)

    log.info('Scan complete')
    log.info('Success: {:d} Error: {:d} Empty: {:d} Total: {:d}'.format(
        len(success_image_ids),
        len(err_image_ids),
        len(empty_image_ids),
        len(image_ids)))

    for image_id in empty_image_ids:
        log.error('Empty annotation in image: {}'.format(image_id))
    for image_id in err_image_ids:
        log.error('No annotation in image: {}'.format(image_id))

    counts_array = np.vstack(counts_array)
    log.info('Statistics')
    stat_count(counts_array)

    cocoqa = COCOQA(base_dir=args.cocoqa_datadir, set_name=args.set)
    cocoqa_dist = stat_cocoqa(cocoqa)
    log.info('COCO-QA dist: {}'.format(cocoqa_dist))

    # Apply COCO-QA distribution and filter questions.
    keep = apply_distribution(questions, cocoqa_dist)
    questions = questions[keep]
    log.info('Generated {:d} questions.'.format(len(questions)))

    if args.output:
        dirname = os.path.dirname(args.output)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        log.info('Writing output to {}'.format(os.path.abspath(args.output)))
        fout = ShardedFile(args.output, num_shards=1)
        with ShardedFileWriter(fout, num_objects=len(keep)) as writer:
            for k in xrange(len(keep)):
                questions[k]['category'] = np.array(
                    [questions[k]['category']], dtype='int16')
                questions[k]['number'] = np.array(
                    [questions[k]['number']], dtype='int16')
                writer.write(questions[k])

    pass
