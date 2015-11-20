from dataset import Dataset
from pycocotools.coco import COCO
import os.path

train_annotation_fname = 'annotations/instances_train2014.json'
valid_annotation_fname = 'annotations/instances_val2014.json'
train_image_id_fname = '/ais/gobi3/u/mren/data/mscoco/imgids_train.txt'
train_image_id_fname = '/ais/gobi3/u/mren/data/mscoco/imgids_valid.txt'

class Mscoco(Dataset):
    """
    MS-COCO API
    """

    def __init__(self, base_dir, set_name='train'):
        if set_name == 'train':
            self._annotation_fname = os.path.join(
                base_dir, train_annotation_fname)
        elif set_name == 'valid':
            self._annotation_fname = os.path.join(
                base_dir, valid_annotation_fname)
        else:
            raise Exception('Set name {} not found'.format(set_name))
        self._coco = COCO(self._annotation_fname)
        pass

    def get_cat_dict(self):
        """
        """
        cat_dict = {}
        for cat_id in self._coco.cats.iterkeys():
            cat_dict[cat_id] = self._coco.cats[cat_id]['name']
        return cat_dict

    def get_image_ids(self):
        """
        """
        return coco.getImgIds()

    def to_zero_id(self):
        pass

    def to_original_id(self):
        pass

    def get_image_annotations(image_id):
        return coco.imgToAnns[image_id]
