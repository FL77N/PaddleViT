import numpy as np
import paddle
from yacs.config import CfgNode as CN
# from ppdet.modeling.proposal_generator.anchor_generator import AnchorGenerator
# from ppdet.modeling.proposal_generator.proposal_generator import ProposalGenerator
# from ppdet.modeling.proposal_generator.target_layer import RPNTargetAssign
# from ppdet.modeling.proposal_generator.rpn_head import RPNHead as PPdetRPNHead


from rpn_head import RPNHead
from rpn_head_ppdet import AnchorGenerator, ProposalGenerator, RPNTargetAssign
from rpn_head_ppdet import RPNHead as PPdetRPNHead

np.random.seed(101010)
paddle.seed(101010)

config = CN()
config.FPN = CN()
config.RPN = CN()

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

gt_boxes1 = paddle.to_tensor([[12, 32, 123, 100],
                              [32, 21, 143, 165],
                              [100, 231, 432, 543],
                              [432, 213, 542, 321],
                              [233, 432, 321, 521]]).astype("float32")

gt_boxes2 = paddle.to_tensor([[2, 3, 54, 63],
                              [12, 43, 231, 182],
                              [50, 20, 431, 321],
                              [321, 432, 542, 654],
                              [211, 332, 321, 321],
                              [22, 32, 93, 95]]).astype("float32")

is_crowd1 = paddle.to_tensor([[0], [0], [1], [0], [0]])
is_crowd2 = paddle.to_tensor([[0], [0], [0], [1], [0], [0]])

imgs_shape = paddle.to_tensor([[800, 800],
                               [800, 800]]).astype("float32")

inputs = {}
inputs["gt_boxes"] = [gt_boxes1, gt_boxes2]
inputs["gt_bbox"] = [gt_boxes1, gt_boxes2]
inputs["is_crowd"] = [is_crowd1, is_crowd2]
inputs["imgs_shape"] = imgs_shape
inputs["im_shape"] = imgs_shape


feat1 = paddle.rand([2, 256, 200, 200]).astype("float32")
feat2 = paddle.rand([2, 256, 100, 100]).astype("float32")
feat3 = paddle.rand([2, 256, 50, 50]).astype("float32")
feat4 = paddle.rand([2, 256, 25, 25]).astype("float32")

feats = [feat1, feat2, feat3, feat4]

rpn_head = RPNHead(config)
rpn_head.eval()
outs = rpn_head.predict(feats)

out = rpn_head(feats, inputs)
print("ppvit:", '\n', out)

ppdet_anchor_gen = AnchorGenerator(
    anchor_sizes = config.RPN.ANCHOR_SIZE,
    aspect_ratios = config.RPN.ASPECT_RATIOS,
    strides = config.RPN.STRIDES,
    variance=[1.0, 1.0, 1.0, 1.0],
    offset=0.
    )

ppdet_proposal_gen_train = ProposalGenerator(
    pre_nms_top_n = config.RPN.PRE_NMS_TOP_N_TRAIN,
    post_nms_top_n = config.RPN.POST_NMS_TOP_N_TRAIN,
    nms_thresh = config.RPN.NMS_THRESH,
    min_size = config.RPN.MIN_SIZE,
    eta=1.,
    topk_after_collect = config.RPN.TOPK_AFTER_COLLECT
    )

ppdet_proposal_gen_test = ProposalGenerator(
    pre_nms_top_n = config.RPN.PRE_NMS_TOP_N_TEST,
    post_nms_top_n = config.RPN.POST_NMS_TOP_N_TEST,
    nms_thresh = config.RPN.NMS_THRESH,
    min_size = config.RPN.MIN_SIZE,
    eta=1.,
    topk_after_collect = config.RPN.TOPK_AFTER_COLLECT
    )

ppdet_tgt_assign = RPNTargetAssign(
    batch_size_per_im = config.RPN.BATCH_SIZE_PER_IMG,
    fg_fraction = config.RPN.POSITIVE_FRACTION,
    positive_overlap = config.RPN.POSITIVE_THRESH,
    negative_overlap = config.RPN.NEGATIVE_THRESH,
    ignore_thresh = -1.,
    use_random = True
    )

ppdet_rpn_head = PPdetRPNHead(
    anchor_generator = ppdet_anchor_gen,
    rpn_target_assign = ppdet_tgt_assign,
    train_proposal = ppdet_proposal_gen_train,
    test_proposal = ppdet_proposal_gen_test,
    in_channel = 256
    )
weight1 = rpn_head.conv.weight
weight2 = rpn_head.objectness_logits.weight
weight3 = rpn_head.anchor_deltas.weight
ppdet_rpn_head.rpn_feat.rpn_conv.weight.set_value(weight1)
ppdet_rpn_head.rpn_rois_score.weight.set_value(weight2)
ppdet_rpn_head.rpn_rois_delta.weight.set_value(weight3)

# rpn_feats = ppdet_rpn_head.rpn_feat(feats)
# for rpn_feat in rpn_feats:
#     rrs = ppdet_rpn_head.rpn_rois_score(rpn_feat)
#     rrd = ppdet_rpn_head.rpn_rois_delta(rpn_feat)
#     print(rrs)
#     break
    # print(rrd)
ppdet_rpn_head.eval()
out = ppdet_rpn_head(feats, inputs)
print("ppdet:", '\n', out)


