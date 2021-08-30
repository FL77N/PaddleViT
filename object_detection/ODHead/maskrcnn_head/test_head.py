import numpy as np
import paddle
from yacs.config import CfgNode as CN

from rpn_head import RPNHead
from roi_head import RoIHead

np.random.seed(101010)
paddle.seed(101010)

config = CN()
config.FPN = CN()
config.RPN = CN()
config.ROI = CN()
config.ROI.BOX_HEAD = CN()

config.FPN.OUT_CHANNELS = 256

config.RPN.ANCHOR_SIZE = [[32], [64], [128], [256], [512]]
config.RPN.ASPECT_RATIOS = [0.5, 1.0, 2.0]
config.RPN.STRIDES = [4, 8, 16, 32, 64]
config.RPN.OFFSET = 0.0
config.RPN.PRE_NMS_TOP_N_TRAIN = 2000
config.RPN.POST_NMS_TOP_N_TRAIN = 1000
config.RPN.PRE_NMS_TOP_N_TEST = 1000
config.RPN.POST_NMS_TOP_N_TEST = 1000
config.RPN.NMS_THRESH = 0.7
config.RPN.MIN_SIZE = 0.0
config.RPN.TOPK_AFTER_COLLECT = True
config.RPN.POSITIVE_THRESH = 0.7
config.RPN.NEGATIVE_THRESH = 0.3
config.RPN.BATCH_SIZE_PER_IMG = 256
config.RPN.POSITIVE_FRACTION = 0.5
config.RPN.LOW_QUALITY_MATCHES = True

config.ROI.SCORE_THRESH_INFER = 0.05
config.ROI.NMS_THRESH_INFER = 0.5
config.ROI.NMS_KEEP_TOPK_INFER =100
config.ROI.NUM_ClASSES = 80
config.ROI.POSITIVE_THRESH = 0.5
config.ROI.NEGATIVE_THRESH = 0.5
config.ROI.BATCH_SIZE_PER_IMG = 512
config.ROI.POSITIVE_FRACTION = 0.25
config.ROI.LOW_QUALITY_MATCHES = True
config.ROI.BOX_HEAD.REG_WEIGHTS = [10.0, 10.0, 5.0, 5.0]
config.ROI.BOX_HEAD.NUM_CONV = 0
config.ROI.BOX_HEAD.CONV_DIM = 256
config.ROI.BOX_HEAD.NUM_FC = 2
config.ROI.BOX_HEAD.FC_DIM = 1024
config.ROI.SCALES = [1./4., 1./8., 1./16., 1./32., 1./64.]
config.ROI.ALIGN_OUTPUT_SIZE = 7
config.ROI.SAMPLING_RATIO = 0
config.ROI.CANONICAL_BOX_SIZE = 224
config.ROI.CANONICAL_LEVEL = 4
config.ROI.MIN_LEVEL = 0
config.ROI.MAX_LEVEL = 3
config.ROI.ALIGNED = True

rpnhead = RPNHead(config)
roihead = RoIHead(config)

gt_boxes1 = paddle.to_tensor([[12, 32, 123, 100],
                              [32, 21, 143, 165],
                              [100, 231, 432, 543],
                              [432, 213, 542, 321],
                              [233, 432, 321, 521]]).astype("float32")

gt_boxes2 = paddle.to_tensor([[2, 3, 54, 63],
                              [12, 43, 231, 182],
                              [50, 20, 431, 321],
                              [321, 432, 542, 654],
                              [211, 180, 321, 321],
                              [22, 32, 93, 95]]).astype("float32")

gt_cls1 = paddle.to_tensor([5, 50, 78, 23, 34]).astype("int32")
gt_cls2 = paddle.to_tensor([2, 23, 51, 25, 36, 76]).astype("int32")

imgs_shape = paddle.to_tensor([[800, 800],
                               [800, 800]]).astype("float32")

is_crowd1 = paddle.to_tensor([[0], [0], [1], [0], [0]])
is_crowd2 = paddle.to_tensor([[0], [0], [0], [1], [0], [0]])

scale_factor_wh = paddle.to_tensor([[1.0, 1.0],
                                    [1.0, 1.0]])

inputs = {}
inputs["gt_classes"] = [gt_cls1, gt_cls2]
inputs["gt_boxes"] = [gt_boxes1, gt_boxes2]
inputs["is_crowd"] = [is_crowd1, is_crowd2]
inputs["imgs_shape"] = imgs_shape
inputs["scale_factor_wh"] = scale_factor_wh

feat1 = paddle.rand([2, 256, 200, 200]).astype("float32")
feat2 = paddle.rand([2, 256, 100, 100]).astype("float32")
feat3 = paddle.rand([2, 256, 50, 50]).astype("float32")
feat4 = paddle.rand([2, 256, 25, 25]).astype("float32")

feats = [feat1, feat2, feat3, feat4]
rpnhead.eval()
roihead.eval()
out = rpnhead(feats, inputs)
proposals = out[0]
final_out = roihead(feats, proposals, inputs)

print(final_out)

