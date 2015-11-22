from pycocotools.coco import COCO
import os.path

train_annotation_fname = 'annotations/instances_train2014.json'
valid_annotation_fname = 'annotations/instances_val2014.json'
train_image_id_fname = '/ais/gobi3/u/mren/data/mscoco/imgids_train.txt'
train_image_id_fname = '/ais/gobi3/u/mren/data/mscoco/imgids_valid.txt'

class MSCOCO(object):
    """MS-COCO API"""

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
        """Returns a dictionary maps from category ID to category name."""
        cat_dict = {}
        for cat_id in self._coco.cats.iterkeys():
            cat_dict[cat_id] = self._coco.cats[cat_id]['name']

        return cat_dict

    def get_cat_list(self):
        """Returns a list maps from integer ID to category name."""
        # Add background class to be consistent with Fast-RCNN encoding.
        cat_dict = self.get_cat_dict()
        cat_list = ['__background__']
        for cat in cat_dict.itervalues():
            cat_list.append(cat)

        return cat_list

    def get_cat_list_reverse(self):
        """Returns a dictionary maps from category ID to integer ID."""
        cat_dict = self.get_cat_dict()
        r = {}
        for i, key in enumerate(cat_dict.iterkeys()):
            r[key] = i

        return r

    def get_image_ids(self):
        """
        """
        return self._coco.getImgIds()

    def to_zero_id(self):
        pass

    def to_original_id(self):
        pass

    def get_image_annotations(self, image_id):
        """Get annotations of an image
        
        Args:
            image_id: string, image ID.
        Returns:
            annotations: list, list of annotation dictionaries.
        """
        if type(image_id) is str:
            image_id = int(image_id)
        if image_id in self._coco.imgToAnns:
            return self._coco.imgToAnns[image_id]
        else:
            return None
