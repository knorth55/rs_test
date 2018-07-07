import numpy as np

from chainer.backends import cuda
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
import warnings


class SSDPredictor(object):

    def __init__(
            self, model='ssd300', pretrained_model='voc0712',
            gpu=-1, score_thresh=0.3
    ):
        self.label_names = voc_bbox_label_names
        if model == 'ssd300':
            model_class = SSD300
        elif model == 'ssd512':
            model_class = SSD512
        else:
            warnings.warn('no model class: {}'.format(model))
        self.model = model_class(
            n_fg_class=len(self.label_names),
            pretrained_model=pretrained_model)
        self.model.score_thresh = score_thresh
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def predict(self, img):
        img = img[:, :, ::-1].transpose((2, 0, 1))
        imgs = img[None]
        bboxes, labels, scores = self.model.predict(imgs)
        bbox, label, score = bboxes[0], labels[0], scores[0]
        bbox = np.round(bbox).astype(np.int32)
        return bbox, label, score
