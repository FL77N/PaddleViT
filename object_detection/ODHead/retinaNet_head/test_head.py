import numpy as np
import paddle
from yacs.config import CfgNode as CN

from retinanet_head import RetinaNetHead

np.random.seed(101010)
paddle.seed(101010)

config = CN()
config.RETINANET = CN()

config.RETINANET.NUM_CONVS = 4
config.RETINANET.INPUT_CHANNELS = 256
config.RETINANET.NORM = ""
config.RETINANET.PRIOR_PROB = 0.01
config.RETINANET.NUM_CLASSES = 80
config.RETINANET.FOCAL_LOSS_ALPHA = 0.25
config.RETINANET.FOCAL_LOSS_GAMMA = 2
config.RETINANET.SMOOTHL1_LOSS_DELTA = 0
config.RETINANET.POSITIVE_THRESH = 0.5
config.RETINANET.NEGATIVE_THRESH = 0.4
config.RETINANET.ALLOW_LOW_QUALITY = True
config.RETINANET.WEIGHTS = [1.0, 1.0, 1.0, 1.0]
config.RETINANET.SCORE_THRESH = 0.05
config.RETINANET.KEEP_TOPK = 100
config.RETINANET.NMS_TOPK = 1000
config.RETINANET.NMS_THRESH = 0.5
config.RETINANET.ANCHOR_SIZE = [[x, x * 2**(1.0/3), x * 2**(2.0/3)] for x in [32, 64, 128, 256, 512 ]]
config.RETINANET.ASPECT_RATIOS = [0.5, 1.0, 2.0]
config.RETINANET.STRIDES = [8.0, 16.0, 32.0, 64.0, 128.0]
config.RETINANET.OFFSET = 0

head = RetinaNetHead(config)

gt_boxes1 = paddle.to_tensor([[12, 32, 123, 100],
                              [32, 21, 143, 165],
                              [100, 231, 432, 543],
                              [432, 213, 542, 321],
                              [233, 432, 321, 521]]).astype("float32")

gt_boxes2 = paddle.to_tensor([[2, 3, 54, 63],
                              [12, 43, 231, 182],
                              [50, 20, 431, 321],
                              [321, 432, 542, 630],
                              [211, 180, 321, 321],
                              [22, 32, 93, 95]]).astype("float32")

gt_cls1 = paddle.to_tensor([5, 50, 78, 23, 34]).astype("int32")
gt_cls2 = paddle.to_tensor([2, 23, 51, 25, 36, 76]).astype("int32")

imgs_shape = paddle.to_tensor([[640, 640],
                               [640, 640]]).astype("float32")

is_crowd1 = paddle.to_tensor([[0], [0], [0], [0], [0]])
is_crowd2 = paddle.to_tensor([[0], [0], [0], [0], [0], [0]])

scale_factor_wh = paddle.to_tensor([[1.0, 1.0],
                                    [1.0, 1.0]])

inputs = {}
inputs["gt_classes"] = [gt_cls1, gt_cls2]
inputs["gt_boxes"] = [gt_boxes1, gt_boxes2]
inputs["is_crowd"] = [is_crowd1, is_crowd2]
inputs["imgs_shape"] = imgs_shape
inputs["scale_factor_wh"] = scale_factor_wh

feat1 = paddle.rand([2, 256, 80, 80]).astype("float32")
feat2 = paddle.rand([2, 256, 40, 40]).astype("float32")
feat3 = paddle.rand([2, 256, 20, 20]).astype("float32")
feat4 = paddle.rand([2, 256, 10, 10]).astype("float32")
feat5 = paddle.rand([2, 256, 5, 5]).astype("float32")

feats = [feat1, feat2, feat3, feat4, feat5]

head.training = False
out = head(feats, inputs)

print(out)