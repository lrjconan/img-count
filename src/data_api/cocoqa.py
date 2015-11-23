import sys
sys.path.insert(0, '../')

from utils import list_reader
from utils import logger
import numpy as np
import os.path

question_fname = 'questions.txt'
answer_fname = 'answers.txt'
question_type_fname = 'types.txt'
image_id_fname = 'img_ids.txt'
train_dirname = 'train'
valid_dirname = 'valid'
train_mscoco_image_id_fname = '/ais/gobi3/u/mren/data/mscoco/imgids_train.txt'
valid_mscoco_image_id_fname = '/ais/gobi3/u/mren/data/mscoco/imgids_valid.txt'

log = logger.get()


class COCOQA(object):
    """
    COCO-QA API
    """

    def __init__(self, base_dir, set_name='train'):
        if set_name == 'train':
            self._dirname = train_dirname
            self._mscoco_image_id_fname = train_mscoco_image_id_fname
        elif set_name == 'valid':
            self._dirname = valid_dirname
            self._mscoco_image_id_fname = valid_mscoco_image_id_fname
        else:
            raise Exception('Set name {} not found'.format(set_name))

        self.base_dir = base_dir
        self._question_fname = os.path.join(
            base_dir, self._dirname, question_fname)
        self._answer_fname = os.path.join(
            base_dir, self._dirname, answer_fname)
        self._question_type_fname = os.path.join(
            base_dir, self._dirname, question_type_fname)
        self._image_id_fname = os.path.join(
            base_dir, self._dirname, image_id_fname)

        # Reading files.
        log.info('Reading files')
        self._questions = list_reader.read_file_list(self._question_fname)
        self._answers = list_reader.read_file_list(self._answer_fname)
        question_type_str = list_reader.read_file_list(
            self._question_type_fname)
        self._question_types = [int(qt) for qt in question_type_str]
        self._image_ids = list_reader.read_file_list(self._image_id_fname)
        self._mscoco_image_ids = list_reader.read_file_list(
            self._mscoco_image_id_fname)

        # Build dictionaries.
        log.info('Building dictionaries')
        question_dict = self._build_vocab_dict(self._questions, keystart=1)
        self._question_dict = question_dict['dict']
        self._question_inv_dict = question_dict['inv_dict']
        answer_dict = self._build_vocab_dict(self._answers, keystart=0)
        self._answer_dict = answer_dict['dict']
        self._answer_inv_dict = answer_dict['inv_dict']
        self._question_type_dict = {
            'object': 0,
            'number': 1,
            'color': 2,
            'location': 3
        }
        self._question_type_inv_dict = [
            'object', 'number', 'color', 'location']
        image_id_dict = self._reindex_image_ids(
            self._mscoco_image_ids, keystart=1)
        self._image_id_dict = image_id_dict['dict']
        self._image_id_inv_dict = image_id_dict['inv_dict']

        # Encode dataset.
        self._question_max_len = self._find_max_len(self._questions)
        self._encoded_questions = self._encode_questions(
            self._questions, self._question_dict, maxlen=self._question_max_len)
        self._encoded_answers = self._encode_answers(
            self._answers, self._answer_dict)
        self._encoded_image_ids = self._encode_image_ids(
            self._image_ids, self._image_id_dict)

        pass

    @staticmethod
    def _build_vocab_dict(lines, keystart, pr=False):
        """Build vocabulary dictionary
        """
        # From word to number.
        word_dict = {}
        # From number to word, numbers need to minus one to convert to list
        # indices.
        word_array = []
        # Word frequency
        word_freq = []
        # if key is 1-based, then 0 is reserved for sentence end.
        key = keystart
        total = 0

        for i in xrange(len(lines)):
            line = lines[i].replace(',', '')
            words = line.split(' ')
            for j in xrange(len(words)):
                if not word_dict.has_key(words[j]):
                    word_dict[words[j]] = key
                    word_array.append(words[j])
                    word_freq.append(1)
                    key += 1
                else:
                    k = word_dict[words[j]]
                    word_freq[k - keystart] += 1
                total += 1
        word_dict['UNK'] = key
        word_array.append('UNK')
        sorted_x = sorted(range(len(word_freq)),
                          key=lambda k: word_freq[k], reverse=True)
        if pr:
            summ = 0
            for x in sorted_x:
                log.info('{}: {}'.format(word_array[x], word_freq[x]))
                summ += word_freq[x]
            med = summ / 2
            medsumm = 0
            for x in sorted_x:
                if medsumm > med:
                    break
                medsumm += word_freq[x]
            log.info('Median: {}',format(word_array[x]))
            log.info('Median freq: {}'.format(word_freq[x]))
            log.info('Median freq %: {}'.format(word_freq[x] / float(total)))
            log.info('Dictionary length: {}'.format(len(word_dict)))
            log.info('Total: {}'.format(total))
        return {
            'dict': word_dict,
            'inv_dict': word_array,
            'freq': word_freq
        }

    @staticmethod
    def _reindex_image_ids(image_ids, keystart):
        image_id_list = []
        image_id_dict = {}
        key = keystart
        for image_id in image_ids:
            image_id_list.append(image_id)
            image_id_dict[image_id] = key
            key += 1
        return {
            'dict': image_id_dict,
            'inv_dict': image_id_list
        }

    @staticmethod
    def _encode_answers(answers, ansdict):
        ansids = []
        for ans in answers:
            if ansdict.has_key(ans):
                ansids.append(ansdict[ans])
            else:
                ansids.append(ansdict['UNK'])
        return np.array(ansids, dtype=int).reshape(len(ansids), 1)

    @staticmethod
    def _encode_questions(questions, worddict, maxlen):
        wordslist = []
        for q in questions:
            words = q.split(' ')
            wordslist.append(words)
            if len(words) > maxlen:
                maxlen = len(words)
        result = np.zeros((len(questions), maxlen, 1), dtype=int)
        for i, words in enumerate(wordslist):
            for j, w in enumerate(words):
                if worddict.has_key(w):
                    result[i, j, 0] = worddict[w]
                else:
                    result[i, j, 0] = worddict['UNK']
        return result

    @staticmethod
    def _encode_image_ids(image_ids, image_id_dict):
        return [image_id_dict[i] for i in image_ids]

    @staticmethod
    def _find_max_len(questions):
        maxlen = 0
        sumlen = 0
        for q in questions:
            words = q.split(' ')
            sumlen += len(words)
            if len(words) > maxlen:
                maxlen = len(words)
        log.info('Max question length: {}'.format(maxlen))
        log.info('Mean question len: {}'.format(sumlen / float(len(questions))))
        return maxlen

    def get_question_vocab_dict(self):
        return self._question_dict

    def get_question_vocab_inv_dict(self):
        return self._question_inv_dict

    def get_answer_vocab_dict(self):
        return self._answer_dict

    def get_answer_vocab_inv_dict(self):
        return self._answer_inv_dict

    def get_encoded_questions(self):
        return self._encoded_questions

    def get_encoded_answers(self):
        return self._encoded_answers

    def get_encoded_image_ids(self):
        return None

    def get_question_types(self):
        return self._question_types

    def get_question_type_dict(self):
        return self._question_type_dict

    def get_image_ids(self):
        return self._image_ids

    def get_image_id_dict(self):
        return self._image_id_dict

    def get_image_id_inv_dict(self):
        return self._image_id_inv_dict
