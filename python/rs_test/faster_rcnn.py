from chainer.backends import cuda
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16


class FasterRCNNPredictor(object):

    def __init__(self, pretrained_model='voc07', gpu=-1):
        self.model = FasterRCNNVGG16(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=pretrained_model)
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def predict(self, img):
        img = img[:, :, ::-1].transpose((2, 0, 1))
        imgs = img[None]
        if self.gpu >= 0:
            imgs = cuda.to_gpu(imgs)
        bboxes, labels, scores = self.model.predict(imgs)
        bbox, label, score = bboxes[0], labels[0], scores[0]
        return bbox, label, score
