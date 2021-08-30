import sys

import paddle
import paddle.nn as nn

from config import get_config
from resnet import ResNet
from data import build_coco, get_dataloader
from coco_eval import evaluate
sys.path.append("PPViT-od_head/object_detection")
from ODNeck.fpn import FPN, LastLevelMaxPool
from ODHead.maskrcnn_head.rpn_head import RPNHead
from ODHead.maskrcnn_head.roi_head import RoIHead

cfg = get_config()


class SwinTransformerDet(nn.Layer):
    def __init__(self, config):
        super(SwinTransformerDet, self).__init__()
        self.backbone = ResNet()
        self.neck = FPN(
            in_channels=config.FPN.IN_CHANNELS,
            out_channel=config.FPN.OUT_CHANNELS,
            strides=config.FPN.STRIDES,
            use_c5=config.FPN.USE_C5,
            top_block=LastLevelMaxPool()     
        )
        self.rpnhead = RPNHead(config)
        self.roihead = RoIHead(config)

        self.config = config
    
    def forward(self, x, gt):
        feats = self.neck(self.backbone(x.tensors))
        rpn_out = self.rpnhead(feats, gt)

        if self.config.ROI.PAT_GT:
            proposals = []
            for proposal, gt_box in zip(rpn_out[0], gt["gt_boxes"]):
                proposals.append(paddle.concat([proposal, gt_box]))
        else:
            proposals = rpn_out[0]

        final_out = self.roihead(feats, proposals, gt)
        return final_out

weight = paddle.load("/home/aistudio/data/data105863/mask_rcnn_r50_fpn_2x_coco.pdparams")
detector = SwinTransformerDet(cfg)
w_dict = detector.state_dict()

for dk, wv in zip(w_dict.keys(), weight.values()):
    w_dict[dk] = wv

# for dk, wk in zip(w_dict.keys(), weight.keys()):
#     print("dk:", dk)
#     print("wk:", wk)

detector.load_dict(w_dict)


def eval(model, cfg):
    dataset = build_coco("val", cfg.DATA.VAL_DATA_PATH)
    dataloader = get_dataloader(dataset, 1, mode='val', multi_gpu=False)
    model.eval()

    results = {}
    for iteration, (data, tgt) in enumerate(dataloader):
        prediction = model(data, tgt)
        results.update({int(id):pred for id, pred in zip(tgt["image_id"], prediction)})

        if iteration % 100 == 0:
            print("[*] Eval Completed: {}".format(iteration))
    
    print()
    print("------------------------------------- compute res -------------------------------------")
    eval_res = evaluate(dataset, results)
    print(eval_res)


eval(detector, cfg)