import numpy as np
import os.path as osp
import warnings

from chainer.backends import cuda
from chainer_mask_rcnn.functions import roi_align_2d
from chainer_mask_rcnn.models import MaskRCNNResNet
import rospkg

from rs_test.coco_utils import coco_instance_segmentation_label_names
from rs_test.fcis import mask_to_roi_mask


class MaskRCNNPredictor(object):

    def __init__(
            self, model='mask_rcnn_resnet50', pretrained_model='coco',
            gpu=-1, score_thresh=0.3):
        self.label_names = coco_instance_segmentation_label_names
        self.score_thresh = score_thresh
        if model == 'mask_rcnn_resnet50':
            model_class = MaskRCNNResNet
            n_layers = 50
            mean = (123.152, 115.903, 103.063)
            anchor_scales = (2, 4, 8, 16, 32)
            min_size = 800
            max_size = 1333
            roi_size = 7
        else:
            warnings.warn('no model class: {}'.format(model))

        if pretrained_model == 'coco':
            r = rospkg.RosPack()
            pretrained_model = osp.join(
                r.get_path('rs_test'),
                'trained_data/mask_rcnn_resnet50_coco_trained.npz')
        self.model = model_class(
            n_layers=n_layers,
            n_fg_class=len(self.label_names),
            pretrained_model=pretrained_model,
            pooling_func=roi_align_2d,
            mean=mean, anchor_scales=anchor_scales,
            min_size=min_size, max_size=max_size, roi_size=roi_size,
        )
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def predict(self, img):
        img = img[:, :, ::-1].transpose((2, 0, 1))
        imgs = img[None]
        bboxes, masks, labels, scores = self.model.predict(imgs)
        bbox, mask, label, score = bboxes[0], masks[0], labels[0], scores[0]
        indices = score > self.score_thresh
        bbox = bbox[indices]
        mask = mask[indices]
        label = label[indices]
        score = score[indices]

        bbox = np.round(bbox).astype(np.int32)
        mask = (mask * 255).astype(np.uint8)
        roi_mask = mask_to_roi_mask(mask, bbox)
        return roi_mask, bbox, label, score
