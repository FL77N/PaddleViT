import sys

import paddle
import paddle.nn as nn

from config import get_config
from backbone import SwinTransformer
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
        self.backbone = SwinTransformer(config)
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
    
    def forward(self, x, gt=None):
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


def eval(model, cfg):
    dataset = build_coco("val", cfg.DATA.VAL_DATA_PATH)
    dataloader = get_dataloader(dataset, cfg.DATA.BATCH_SIZE_EVAL, mode='val', multi_gpu=False)
    model.eval()

    results = {}
    for iteration, (data, tgt) in enumerate(dataloader):
        prediction = model(data, tgt)
        results.update({int(id):pred for id, pred in zip(tgt["image_id"], prediction)})

        if iteration % 100 == 0:
            print("[*] Eval Completed: {}".format(iteration))
    
    print()
    print("<------------------------------------- compute res ------------------------------------->")
    eval_res = evaluate(dataset, results)
    print(eval_res)


detector = SwinTransformerDet(cfg)
weights = paddle.load(cfg.DATA.WEIGHT_PATH)
detector.load_dict(weights)
eval(detector, cfg)


