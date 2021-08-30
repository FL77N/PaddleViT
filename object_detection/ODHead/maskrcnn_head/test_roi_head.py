import sys
import numpy as np
import paddle
from yacs.config import CfgNode as CN

from rpn_head import RPNHead
from roi_head import RoIHead
from head_ppdet import AnchorGenerator, ProposalGenerator, RPNTargetAssign, BBoxHead, TwoFCHead, RoIAlign
from head_ppdet import RPNHead as PPdetRPNHead

sys.path.append("PPViT-od_head/object_detection/SwinTransformer")
from config import get_config
from data import build_coco, get_dataloader
from coco_eval import evaluate
from resnet import ResNet, FPN
sys.path.append("PPViT-od_head/object_detection/ODNeck")
from fpn import FPN as ViTFPN
from fpn import LastLevelMaxPool

cfg = get_config()

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
                              [321, 432, 542, 640],
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
inputs["im_shape"] = imgs_shape
inputs["scale_factor_wh"] = scale_factor_wh
inputs["orig_size"] = imgs_shape

imgs = paddle.rand([2, 3, 640, 640]).astype("float32")
feat1 = paddle.rand([2, 256, 160, 160]).astype("float32")
feat2 = paddle.rand([2, 256, 80, 80]).astype("float32")
feat3 = paddle.rand([2, 256, 40, 40]).astype("float32")
feat4 = paddle.rand([2, 256, 20, 20]).astype("float32")
feat5 = paddle.rand([2, 256, 10, 10]).astype("float32")

# feats = [feat1, feat2, feat3, feat4, feat5]

backbone_weights = {}
neck_weights = {}
head_weights = {}
weights = paddle.load("/home/aistudio/data/data105863/mask_rcnn_r50_fpn_2x_coco.pdparams")

for k, v in weights.items():
    name_list = k.split('.')
    if "backbone" in name_list:
        backbone_weights[k] = v

for k, v in weights.items():
    name_list = k.split('.')
    if "neck" in name_list:
        neck_weights[k] = v

for k, v in weights.items():
    name_list = k.split('.')
    if "rpn_head" in name_list or "bbox_head" in name_list:
        head_weights[k] = v

res_w = {}
backbone = ResNet()
for k, v in zip(backbone.state_dict().keys(), list(backbone_weights.values())):
    res_w[k] = v
backbone.load_dict(res_w)

vitfpn_w = {}
neck1 = ViTFPN(
    in_channels=[256, 512, 1024, 2048],
    out_channel=256,
    strides=[4, 8, 16, 32],
    use_c5=False,
    top_block=LastLevelMaxPool()     
)

for k, v in zip(neck1.state_dict().keys(), list(neck_weights.values())):
    vitfpn_w[k] = v
neck1.load_dict(vitfpn_w)

rpn_w_1 = {}
roi_w_1 = {}
for k, v in zip(rpnhead.state_dict().keys(), list(head_weights.values())[:6]):
    rpn_w_1[k] = v
for k, v in zip(roihead.state_dict().keys(), list(head_weights.values())[6:]):
    roi_w_1[k] = v

rpnhead.load_dict(rpn_w_1)
roihead.load_dict(roi_w_1)
backbone.eval()
neck1.eval()
rpnhead.eval()
roihead.eval()
feats = neck1(backbone(imgs))
rpn_out1 = rpnhead(feats, inputs)
# print("ppvit rpn:", '\n', rpn_out1)
out1 = roihead(feats, rpn_out1[0], inputs)
# print("ppvit roi:", '\n', out1)

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

roi_align = RoIAlign(
    resolution=7,
    spatial_scale=config.ROI.SCALES,
    sampling_ratio=0,
    aligned=True
)

head = TwoFCHead(
    in_channel=256, 
    out_channel=1024, 
    resolution=7
)

ppdet_roi_head = BBoxHead(
    head=head,
    in_channel=1024,
    roi_extractor=roi_align,
    bbox_assigner='BboxAssigner',
    with_pool=False,
    num_classes=80,
    bbox_weight=[10., 10., 5., 5.],
    bbox_loss=None)

# weight1 = rpnhead.conv.weight
# weight2 = rpnhead.objectness_logits.weight
# weight3 = rpnhead.anchor_deltas.weight
# ppdet_rpn_head.rpn_feat.rpn_conv.weight.set_value(weight1)
# ppdet_rpn_head.rpn_rois_score.weight.set_value(weight2)
# ppdet_rpn_head.rpn_rois_delta.weight.set_value(weight3)

# weight1 = roihead.predictor.forward_net.linear0.weight
# weight2 = roihead.predictor.forward_net.linear1.weight
# weight3 = roihead.predictor.cls_fc.weight
# weight4 = roihead.predictor.reg_fc.weight
# ppdet_roi_head.head.fc6.weight.set_value(weight1)
# ppdet_roi_head.head.fc7.weight.set_value(weight2)
# ppdet_roi_head.bbox_score.weight.set_value(weight3)
# ppdet_roi_head.bbox_delta.weight.set_value(weight4)

# roihead
# ppdet_roi_head
# rpn_feats = ppdet_rpn_head.rpn_feat(feats)
# for rpn_feat in rpn_feats:
#     rrs = ppdet_rpn_head.rpn_rois_score(rpn_feat)
#     rrd = ppdet_rpn_head.rpn_rois_delta(rpn_feat)
#     print(rrs)
#     break
    # print(rrd)


fpn_w = {}
neck2 = FPN(
    in_channels=[256, 512, 1024, 2048],
    out_channel=256,
    spatial_scales=[0.25, 0.125, 0.0625, 0.03125]    
)

for k, v in zip(neck2.state_dict().keys(), list(neck_weights.values())):
    fpn_w[k] = v
neck2.load_dict(fpn_w)

feats = neck1(backbone(imgs))

rpn_w_2 = {}
roi_w_2 = {}
for k, v in zip(ppdet_rpn_head.state_dict().keys(), list(head_weights.values())[:6]):
    rpn_w_2[k] = v
for k, v in zip(ppdet_roi_head.state_dict().keys(), list(head_weights.values())[6:]):
    roi_w_2[k] = v

ppdet_rpn_head.load_dict(rpn_w_2)
ppdet_roi_head.load_dict(roi_w_2)
backbone.eval()
neck2.eval()
ppdet_rpn_head.eval()
ppdet_roi_head.eval()
rpn_out2 = ppdet_rpn_head(feats, inputs)
# print("ppdet rpn:", '\n', rpn_out2)
out2 = ppdet_roi_head(body_feats=feats, rois=rpn_out2[0], rois_num=rpn_out2[1], inputs=inputs)
# print("ppdet roi:", '\n', out2)

for o1, o2 in zip(rpn_out1[0], rpn_out2[0]):
    print("rpn", (o1 == o2).all())

print("roi_s", (out1[1][0] == out2[0][1]).all())
print("roi_r", (out1[1][1] == out2[0][0]).all())

def eval(cfg):
    dataset = build_coco("val", cfg.DATA.VAL_DATA_PATH)
    dataloader = get_dataloader(dataset, 1, mode='val', multi_gpu=False)
    backbone.eval()
    neck1.eval()
    ppdet_rpn_head.eval()
    ppdet_roi_head.eval()
    rpnhead.eval()
    roihead.eval()
    post = RoIHead(cfg)

    results = {}
    for iteration, (data, tgt) in enumerate(dataloader):
        im_shape = paddle.stack(tgt["imgs_shape"])
        scale_factor = paddle.stack(tgt["scale_factor_wh"])[:, ::-1]
        tgt["im_shape"] = im_shape
        feats = neck1(backbone(data.tensors))
        rpnout1 = rpnhead(feats, tgt)
        roiout1 = roihead(feats, rpnout1[0], tgt)
        rpnout2 = ppdet_rpn_head(feats, tgt)
        roiout2 = ppdet_roi_head(body_feats=feats, rois=rpnout2[0], rois_num=rpnout2[1], inputs=tgt)

        for o1, o2 in zip(rpnout1[0], rpnout2[0]):
            print("rpn", (o1 == o2).all())

        print("body_feat", (roiout1[2][0] == roiout2[1][0]).all())
        print("roi_s", (roiout1[1][0] == roiout2[0][1]).all())
        print("roi_r", (roiout1[1][1] == roiout2[0][0]).all())
        prediction = post.post_process([roiout2[0][1], roiout2[0][0]], rpnout2, im_shape, scale_factor)
        results.update({int(id):pred for id, pred in zip(tgt["image_id"], prediction)})

        if iteration % 100 == 0:
            print("[*] Eval Completed: {}".format(iteration))
    
    print()
    print("------------------------------------- compute res -------------------------------------")
    eval_res = evaluate(dataset, results)
    print(eval_res)


eval(cfg)
