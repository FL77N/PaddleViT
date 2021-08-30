'''
    Created on: 05.07.2021
    @Author: feizzhang
'''
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierNormal, XavierUniform
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype

def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def bbox_overlaps(boxes1, boxes2):
    """
    Calculate overlaps between boxes1 and boxes2
    Args:
        boxes1 (Tensor): boxes with shape [M, 4]
        boxes2 (Tensor): boxes with shape [N, 4]
    Return:
        overlaps (Tensor): overlaps between boxes1 and boxes2 with shape [M, N]
    """
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    if M * N == 0:
        return paddle.zeros([M, N], dtype='float32')
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)

    xy_max = paddle.minimum(
        paddle.unsqueeze(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = paddle.maximum(
        paddle.unsqueeze(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(min=0)
    inter = width_height.prod(axis=2)

    overlaps = paddle.where(inter > 0, inter /
                            (paddle.unsqueeze(area1, 1) + area2 - inter),
                            paddle.zeros_like(inter))
    return overlaps


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * paddle.log(tgt_w / src_w)
    dh = wh * paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1)
    return deltas


def rpn_anchor_target(anchors,
                      gt_boxes,
                      rpn_batch_size_per_im,
                      rpn_positive_overlap,
                      rpn_negative_overlap,
                      rpn_fg_fraction,
                      use_random=True,
                      batch_size=1,
                      ignore_thresh=-1,
                      is_crowd=None,
                      weights=[1., 1., 1., 1.]):
    tgt_labels = []
    tgt_bboxes = []
    tgt_deltas = []
    for i in range(batch_size):
        gt_bbox = gt_boxes[i]
        is_crowd_i = is_crowd[i] if is_crowd else None
        # Step1: match anchor and gt_bbox
        matches, match_labels = label_box(
            anchors, gt_bbox, rpn_positive_overlap, rpn_negative_overlap, True,
            ignore_thresh, is_crowd_i)
        # Step2: sample anchor 
        fg_inds, bg_inds = subsample_labels(match_labels, rpn_batch_size_per_im,
                                            rpn_fg_fraction, 0, use_random)
        # Fill with the ignore label (-1), then set positive and negative labels
        labels = paddle.full(match_labels.shape, -1, dtype='int32')
        if bg_inds.shape[0] > 0:
            labels = paddle.scatter(labels, bg_inds, paddle.zeros_like(bg_inds))
        if fg_inds.shape[0] > 0:
            labels = paddle.scatter(labels, fg_inds, paddle.ones_like(fg_inds))
        # Step3: make output  
        if gt_bbox.shape[0] == 0:
            matched_gt_boxes = paddle.zeros([0, 4])
            tgt_delta = paddle.zeros([0, 4])
        else:
            matched_gt_boxes = paddle.gather(gt_bbox, matches)
            tgt_delta = bbox2delta(anchors, matched_gt_boxes, weights)
            matched_gt_boxes.stop_gradient = True
            tgt_delta.stop_gradient = True
        labels.stop_gradient = True
        tgt_labels.append(labels)
        tgt_bboxes.append(matched_gt_boxes)
        tgt_deltas.append(tgt_delta)

    return tgt_labels, tgt_bboxes, tgt_deltas


def label_box(anchors,
              gt_boxes,
              positive_overlap,
              negative_overlap,
              allow_low_quality,
              ignore_thresh,
              is_crowd=None):
    iou = bbox_overlaps(gt_boxes, anchors)
    n_gt = gt_boxes.shape[0]
    if n_gt == 0 or is_crowd is None:
        n_gt_crowd = 0
    else:
        n_gt_crowd = paddle.nonzero(is_crowd).shape[0]
    if iou.shape[0] == 0 or n_gt_crowd == n_gt:
        # No truth, assign everything to background
        default_matches = paddle.full((iou.shape[1], ), 0, dtype='int64')
        default_match_labels = paddle.full((iou.shape[1], ), 0, dtype='int32')
        return default_matches, default_match_labels
    # if ignore_thresh > 0, remove anchor if it is closed to 
    # one of the crowded ground-truth
    if n_gt_crowd > 0:
        N_a = anchors.shape[0]
        ones = paddle.ones([N_a])
        mask = is_crowd * ones

        if ignore_thresh > 0:
            crowd_iou = iou * mask
            valid = (paddle.sum((crowd_iou > ignore_thresh).cast('int32'),
                                axis=0) > 0).cast('float32')
            iou = iou * (1 - valid) - valid

        # ignore the iou between anchor and crowded ground-truth
        iou = iou * (1 - mask) - mask

    matched_vals, matches = paddle.topk(iou, k=1, axis=0)
    match_labels = paddle.full(matches.shape, -1, dtype='int32')
    # set ignored anchor with iou = -1
    neg_cond = paddle.logical_and(matched_vals > -1,
                                  matched_vals < negative_overlap)
    match_labels = paddle.where(neg_cond,
                                paddle.zeros_like(match_labels), match_labels)
    match_labels = paddle.where(matched_vals >= positive_overlap,
                                paddle.ones_like(match_labels), match_labels)

    if allow_low_quality:
        highest_quality_foreach_gt = iou.max(axis=1, keepdim=True)
        pred_inds_with_highest_quality = paddle.logical_and(
            iou > 0, iou == highest_quality_foreach_gt).cast('int32').sum(
                0, keepdim=True)
        match_labels = paddle.where(pred_inds_with_highest_quality > 0,
                                    paddle.ones_like(match_labels),
                                    match_labels)

    matches = matches.flatten()
    match_labels = match_labels.flatten()

    return matches, match_labels


def subsample_labels(labels,
                     num_samples,
                     fg_fraction,
                     bg_label=0,
                     use_random=True):
    positive = paddle.nonzero(
        paddle.logical_and(labels != -1, labels != bg_label))
    negative = paddle.nonzero(labels == bg_label)

    fg_num = int(num_samples * fg_fraction)
    fg_num = min(positive.numel(), fg_num)
    bg_num = num_samples - fg_num
    bg_num = min(negative.numel(), bg_num)
    if fg_num == 0 and bg_num == 0:
        fg_inds = paddle.zeros([0], dtype='int32')
        bg_inds = paddle.zeros([0], dtype='int32')
        return fg_inds, bg_inds

    # randomly select positive and negative examples

    negative = negative.cast('int32').flatten()
    bg_perm = paddle.randperm(negative.numel(), dtype='int32')
    bg_perm = paddle.slice(bg_perm, axes=[0], starts=[0], ends=[bg_num])
    if use_random:
        bg_inds = paddle.gather(negative, bg_perm)
    else:
        bg_inds = paddle.slice(negative, axes=[0], starts=[0], ends=[bg_num])
    if fg_num == 0:
        fg_inds = paddle.zeros([0], dtype='int32')
        return fg_inds, bg_inds

    positive = positive.cast('int32').flatten()
    fg_perm = paddle.randperm(positive.numel(), dtype='int32')
    fg_perm = paddle.slice(fg_perm, axes=[0], starts=[0], ends=[fg_num])
    if use_random:
        fg_inds = paddle.gather(positive, fg_perm)
    else:
        fg_inds = paddle.slice(positive, axes=[0], starts=[0], ends=[fg_num])

    return fg_inds, bg_inds

    
def generate_proposals(scores,
                       bbox_deltas,
                       im_shape,
                       anchors,
                       variances,
                       pre_nms_top_n=6000,
                       post_nms_top_n=1000,
                       nms_thresh=0.5,
                       min_size=0.1,
                       eta=1.0,
                       pixel_offset=False,
                       return_rois_num=False,
                       name=None):
    """
    **Generate proposal Faster-RCNN**
    This operation proposes RoIs according to each box with their
    probability to be a foreground object and 
    the box can be calculated by anchors. Bbox_deltais and scores
    to be an object are the output of RPN. Final proposals
    could be used to train detection net.
    For generating proposals, this operation performs following steps:
    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 4)
    2. Calculate box locations as proposals candidates. 
    3. Clip boxes to image
    4. Remove predicted boxes with small area. 
    5. Apply NMS to get final proposals as output.
    Args:
        scores(Tensor): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas(Tensor): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        im_shape(Tensor): A 2-D Tensor with shape [N, 2] represents H, W, the
            origin image size or input size. The data type can be float32 or 
            float64.
        anchors(Tensor):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances(Tensor): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n(float): Number of total bboxes to be kept per
            image before NMS. The data type must be float32. `6000` by default.
        post_nms_top_n(float): Number of total bboxes to be kept per
            image after NMS. The data type must be float32. `1000` by default.
        nms_thresh(float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size(float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
        eta(float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration.
        return_rois_num(bool): When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's 
            num of each image in one batch. The N is the image's num. For example, the tensor has values [4,5] that represents
            the first image has 4 Rois, the second image has 5 Rois. It only used in rcnn model. 
            'False' by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 
    Returns:
        tuple:
        A tuple with format ``(rpn_rois, rpn_roi_probs)``.
        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
    Examples:
        .. code-block:: python
        
            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            scores = paddle.static.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
            bbox_deltas = paddle.static.data(name='bbox_deltas', shape=[None, 16, 5, 5], dtype='float32')
            im_shape = paddle.static.data(name='im_shape', shape=[None, 2], dtype='float32')
            anchors = paddle.static.data(name='anchors', shape=[None, 5, 4, 4], dtype='float32')
            variances = paddle.static.data(name='variances', shape=[None, 5, 10, 4], dtype='float32')
            rois, roi_probs = ops.generate_proposals(scores, bbox_deltas,
                         im_shape, anchors, variances)
    """
    if in_dygraph_mode():
        assert return_rois_num, "return_rois_num should be True in dygraph mode."
        attrs = ('pre_nms_topN', pre_nms_top_n, 'post_nms_topN', post_nms_top_n,
                 'nms_thresh', nms_thresh, 'min_size', min_size, 'eta', eta,
                 'pixel_offset', pixel_offset)
        rpn_rois, rpn_roi_probs, rpn_rois_num = core.ops.generate_proposals_v2(
            scores, bbox_deltas, im_shape, anchors, variances, *attrs)
        return rpn_rois, rpn_roi_probs, rpn_rois_num


class ProposalGenerator(object):
    """
    Proposal generation module
    For more details, please refer to the document of generate_proposals 
    in ppdet/modeing/ops.py
    Args:
        pre_nms_top_n (int): Number of total bboxes to be kept per
            image before NMS. default 6000
        post_nms_top_n (int): Number of total bboxes to be kept per
            image after NMS. default 1000
        nms_thresh (float): Threshold in NMS. default 0.5
        min_size (flaot): Remove predicted boxes with either height or
             width < min_size. default 0.1
        eta (float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
             `adaptive_threshold = adaptive_threshold * eta` in each iteration.
             default 1.
        topk_after_collect (bool): whether to adopt topk after batch 
             collection. If topk_after_collect is true, box filter will not be 
             used after NMS at each image in proposal generation. default false
    """

    def __init__(self,
                 pre_nms_top_n=12000,
                 post_nms_top_n=2000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.,
                 topk_after_collect=False):
        super(ProposalGenerator, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.topk_after_collect = topk_after_collect

    def __call__(self, scores, bbox_deltas, anchors, im_shape):

        top_n = self.pre_nms_top_n if self.topk_after_collect else self.post_nms_top_n
        variances = paddle.ones_like(anchors)
        rpn_rois, rpn_rois_prob, rpn_rois_num = generate_proposals(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=self.pre_nms_top_n,
            post_nms_top_n=top_n,
            nms_thresh=self.nms_thresh,
            min_size=self.min_size,
            eta=self.eta,
            return_rois_num=True)
        return rpn_rois, rpn_rois_prob, rpn_rois_num, self.post_nms_top_n


class RPNTargetAssign(object):
    """
    RPN targets assignment module
    The assignment consists of three steps:
        1. Match anchor and ground-truth box, label the anchor with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and 
           background
        3. Generate the targets for classification and regression branch
    Args:
        batch_size_per_im (int): Total number of RPN samples per image. 
            default 256
        fg_fraction (float): Fraction of anchors that is labeled
            foreground, default 0.5
        positive_overlap (float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be 
            a foreground sample. default 0.7
        negative_overlap (float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be 
            a background sample. default 0.3
        ignore_thresh(float): Threshold for ignoring the is_crowd ground-truth
            if the value is larger than zero.
        use_random (bool): Use random sampling to choose foreground and 
            background boxes, default true.
    """

    def __init__(self,
                 batch_size_per_im=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 ignore_thresh=-1.,
                 use_random=True):
        super(RPNTargetAssign, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.ignore_thresh = ignore_thresh
        self.use_random = use_random

    def __call__(self, inputs, anchors):
        """
        inputs: ground-truth instances.
        anchor_box (Tensor): [num_anchors, 4], num_anchors are all anchors in all feature maps.
        """
        gt_boxes = inputs['gt_bbox']
        is_crowd = inputs.get('is_crowd', None)
        batch_size = len(gt_boxes)
        tgt_labels, tgt_bboxes, tgt_deltas = rpn_anchor_target(
            anchors, gt_boxes, self.batch_size_per_im, self.positive_overlap,
            self.negative_overlap, self.fg_fraction, self.use_random,
            batch_size, self.ignore_thresh, is_crowd)
        norm = self.batch_size_per_im * batch_size

        return tgt_labels, tgt_bboxes, tgt_deltas, norm


class AnchorGenerator(nn.Layer):
    """
    Generate anchors according to the feature maps
    Args:
        anchor_sizes (list[float] | list[list[float]]): The anchor sizes at 
            each feature point. list[float] means all feature levels share the 
            same sizes. list[list[float]] means the anchor sizes for 
            each level. The sizes stand for the scale of input size.
        aspect_ratios (list[float] | list[list[float]]): The aspect ratios at
            each feature point. list[float] means all feature levels share the
            same ratios. list[list[float]] means the aspect ratios for
            each level.
        strides (list[float]): The strides of feature maps which generate 
            anchors
        offset (float): The offset of the coordinate of anchors, default 0.
        
    """

    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[16.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.variance = variance
        self.cell_anchors = self._calculate_anchors(len(strides))
        self.offset = offset

    def _broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):  # list[float]
            return [params] * num_features
        if len(params) == 1:
            return list(params) * num_features
        return params

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return paddle.to_tensor(anchors, dtype='float32')

    def _calculate_anchors(self, num_features):
        sizes = self._broadcast_params(self.anchor_sizes, num_features)
        aspect_ratios = self._broadcast_params(self.aspect_ratios, num_features)
        cell_anchors = [
            self.generate_cell_anchors(s, a)
            for s, a in zip(sizes, aspect_ratios)
        ]
        [
            self.register_buffer(
                t.name, t, persistable=False) for t in cell_anchors
        ]
        return cell_anchors

    def _create_grid_offsets(self, size, stride, offset):
        grid_height, grid_width = size[0], size[1]
        shifts_x = paddle.arange(
            offset * stride, grid_width * stride, step=stride, dtype='float32')
        shifts_y = paddle.arange(
            offset * stride, grid_height * stride, step=stride, dtype='float32')
        shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
        shift_x = paddle.reshape(shift_x, [-1])
        shift_y = paddle.reshape(shift_y, [-1])
        return shift_x, shift_y

    def _grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
                                              self.cell_anchors):
            shift_x, shift_y = self._create_grid_offsets(size, stride,
                                                         self.offset)
            shifts = paddle.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
            shifts = paddle.reshape(shifts, [-1, 1, 4])
            base_anchors = paddle.reshape(base_anchors, [1, -1, 4])

            anchors.append(paddle.reshape(shifts + base_anchors, [-1, 4]))

        return anchors

    def forward(self, input):
        grid_sizes = [paddle.shape(feature_map)[-2:] for feature_map in input]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps

    @property
    def num_anchors(self):
        """
        Returns:
            int: number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                For FPN models, `num_anchors` on every feature map is the same.
        """
        return len(self.cell_anchors[0])


class RPNFeat(nn.Layer):
    """
    Feature extraction in RPN head
    Args:
        in_channel (int): Input channel
        out_channel (int): Output channel
    """

    def __init__(self, in_channel=1024, out_channel=1024):
        super(RPNFeat, self).__init__()
        # rpn feat is shared with each level
        self.rpn_conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_conv.skip_quant = True

    def forward(self, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(F.relu(self.rpn_conv(feat)))
        return rpn_feats

        
class RPNHead(nn.Layer):
    """
    Region Proposal Network
    Args:
        anchor_generator (dict): configure of anchor generation
        rpn_target_assign (dict): configure of rpn targets assignment
        train_proposal (dict): configure of proposals generation
            at the stage of training
        test_proposal (dict): configure of proposals generation
            at the stage of prediction
        in_channel (int): channel of input feature maps which can be
            derived by from_config
    """

    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_proposal=ProposalGenerator(12000, 2000).__dict__,
                 test_proposal=ProposalGenerator().__dict__,
                 in_channel=1024):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = ProposalGenerator(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = ProposalGenerator(**test_proposal)

        num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(in_channel, in_channel)
        # rpn head is shared with each level
        # rpn roi classification scores
        self.rpn_rois_score = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_rois_score.skip_quant = True

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = nn.Conv2D(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_rois_delta.skip_quant = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        # FPN share same rpn head
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, feats, inputs):
        rpn_feats = self.rpn_feat(feats)
        scores = []
        deltas = []

        for rpn_feat in rpn_feats:
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)

        anchors = self.anchor_generator(rpn_feats)

        # TODO: Fix batch_size > 1 when testing.
        if self.training:
            batch_size = inputs['im_shape'].shape[0]
        else:
            batch_size = inputs['im_shape'].shape[0]

        rois, rois_num = self._gen_proposal(scores, deltas, anchors, inputs,
                                            batch_size)
        if self.training:
            loss = self.get_loss(scores, deltas, anchors, inputs)
            return rois, rois_num, loss
        else:
            return rois, rois_num, None

    def _gen_proposal(self, scores, bbox_deltas, anchors, inputs, batch_size):
        """
        scores (list[Tensor]): Multi-level scores prediction
        bbox_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info
        """
        prop_gen = self.train_proposal if self.training else self.test_proposal
        im_shape = inputs['im_shape']

        # Collect multi-level proposals for each batch
        # Get 'topk' of them as final output
        bs_rois_collect = []
        bs_rois_num_collect = []

        # Generate proposals for each level and each batch.
        # Discard batch-computing to avoid sorting bbox cross different batches.
        for i in range(batch_size):
            rpn_rois_list = []
            rpn_prob_list = []
            rpn_rois_num_list = []

            for rpn_score, rpn_delta, anchor in zip(scores, bbox_deltas,
                                                    anchors):
                rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n = prop_gen(
                    scores=rpn_score[i:i + 1],
                    bbox_deltas=rpn_delta[i:i + 1],
                    anchors=anchor,
                    im_shape=im_shape[i:i + 1])
                if rpn_rois.shape[0] > 0:
                    rpn_rois_list.append(rpn_rois)
                    rpn_prob_list.append(rpn_rois_prob)
                    rpn_rois_num_list.append(rpn_rois_num)

            if len(scores) > 1:
                rpn_rois = paddle.concat(rpn_rois_list)
                rpn_prob = paddle.concat(rpn_prob_list).flatten()

                if rpn_prob.shape[0] > post_nms_top_n:
                    topk_prob, topk_inds = paddle.topk(rpn_prob, post_nms_top_n)
                    topk_rois = paddle.gather(rpn_rois, topk_inds)
                else:
                    topk_rois = rpn_rois
                    topk_prob = rpn_prob
            else:
                topk_rois = rpn_rois_list[0]
                topk_prob = rpn_prob_list[0].flatten()

            bs_rois_collect.append(topk_rois)
            bs_rois_num_collect.append(paddle.shape(topk_rois)[0])

        bs_rois_num_collect = paddle.concat(bs_rois_num_collect)

        return bs_rois_collect, bs_rois_num_collect

    def get_loss(self, pred_scores, pred_deltas, anchors, inputs):
        """
        pred_scores (list[Tensor]): Multi-level scores prediction
        pred_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_score
        """
        anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]       # [\sum H_i * W_i * A, 4]
        anchors = paddle.concat(anchors)

        scores = [                                                          # [N, A, H, W] -> [N, H * W * A, 1]
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 1)) for v in pred_scores
        ]
        scores = paddle.concat(scores, axis=1)

        deltas = [                                                          # [N, 4*A, H, W] -> [N, H * W * A, 4]                          
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 4)) for v in pred_deltas
        ]
        deltas = paddle.concat(deltas, axis=1)

        score_tgt, bbox_tgt, loc_tgt, norm = self.rpn_target_assign(inputs,
                                                                    anchors)

        scores = paddle.reshape(x=scores, shape=(-1, ))
        deltas = paddle.reshape(x=deltas, shape=(-1, 4))

        score_tgt = paddle.concat(score_tgt)
        score_tgt.stop_gradient = True

        pos_mask = score_tgt == 1
        pos_ind = paddle.nonzero(pos_mask)

        valid_mask = score_tgt >= 0
        valid_ind = paddle.nonzero(valid_mask)

        # cls loss
        if valid_ind.shape[0] == 0:
            loss_rpn_cls = paddle.zeros([1], dtype='float32')
        else:
            score_pred = paddle.gather(scores, valid_ind)
            score_label = paddle.gather(score_tgt, valid_ind).cast('float32')
            score_label.stop_gradient = True
            loss_rpn_cls = F.binary_cross_entropy_with_logits(
                logit=score_pred, label=score_label, reduction="sum")

        # reg loss
        if pos_ind.shape[0] == 0:
            loss_rpn_reg = paddle.zeros([1], dtype='float32')
        else:
            loc_pred = paddle.gather(deltas, pos_ind)
            loc_tgt = paddle.concat(loc_tgt)
            loc_tgt = paddle.gather(loc_tgt, pos_ind)
            loc_tgt.stop_gradient = True
            loss_rpn_reg = paddle.abs(loc_pred - loc_tgt).sum()
        return {
            'loss_rpn_cls': loss_rpn_cls / norm,
            'loss_rpn_reg': loss_rpn_reg / norm
        }


def _to_list(v):
    if not isinstance(v, (list, tuple)):
        return [v]
    return v


def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             pixel_offset=False,
                             rois_num=None,
                             name=None):
    """
    
    **This op only takes LoDTensor as input.** In Feature Pyramid Networks 
    (FPN) models, it is needed to distribute all proposals into different FPN 
    level, with respect to scale of the proposals, the referring scale and the 
    referring level. Besides, to restore the order of proposals, we return an 
    array which indicates the original index of rois in current proposals. 
    To compute FPN level for each roi, the formula is given as follows:
    
    .. math::

        roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}

        level = floor(&\log(\\frac{roi\_scale}{refer\_scale}) + refer\_level)

    where BBoxArea is a function to compute the area of each roi.

    Args:

        fpn_rois(Variable): 2-D Tensor with shape [N, 4] and data type is 
            float32 or float64. The input fpn_rois.
        min_level(int32): The lowest level of FPN layer where the proposals come 
            from.
        max_level(int32): The highest level of FPN layer where the proposals
            come from.
        refer_level(int32): The referring level of FPN layer with specified scale.
        refer_scale(int32): The referring scale of FPN layer with specified level.
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tuple:

        multi_rois(List) : A list of 2-D LoDTensor with shape [M, 4] 
        and data type of float32 and float64. The length is 
        max_level-min_level+1. The proposals in each FPN level.

        restore_ind(Variable): A 2-D Tensor with shape [N, 1], N is 
        the number of total rois. The data type is int32. It is
        used to restore the order of fpn_rois.

        rois_num_per_level(List): A list of 1-D Tensor and each Tensor is 
        the RoIs' number in each image on the corresponding level. The shape 
        is [B] and data type of int32. B is the number of images


    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            fpn_rois = paddle.static.data(
                name='data', shape=[None, 4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = ops.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)
    """
    num_lvl = max_level - min_level + 1

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        attrs = ('min_level', min_level, 'max_level', max_level, 'refer_level',
                 refer_level, 'refer_scale', refer_scale, 'pixel_offset',
                 pixel_offset)
        multi_rois, restore_ind, rois_num_per_level = core.ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)
        return multi_rois, restore_ind, rois_num_per_level


def roi_align(input,
              rois,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              rois_num=None,
              aligned=True,
              name=None):
    """

    Region of interest align (also known as RoI align) is to perform
    bilinear interpolation on inputs of nonuniform sizes to obtain 
    fixed-size feature maps (e.g. 7*7)

    Dividing each region proposal into equal-sized sections with
    the pooled_width and pooled_height. Location remains the origin
    result.

    In each ROI bin, the value of the four regularly sampled locations 
    are computed directly through bilinear interpolation. The output is
    the mean of four locations.
    Thus avoid the misaligned problem. 

    Args:
        input (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height, W is weight. 
            The data type is float32 or float64.
        rois (Tensor): ROIs (Regions of Interest) to pool over.It should be
            a 2-D Tensor or 2-D LoDTensor of shape (num_rois, 4), the lod level is 1. 
            The data type is float32 or float64. Given as [[x1, y1, x2, y2], ...],
            (x1, y1) is the top left coordinates, and (x2, y2) is the bottom right coordinates.
        output_size (int or tuple[int, int]): The pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float32, optional): Multiplicative spatial scale factor to translate ROI coords 
            from their input scale to the scale used when pooling. Default: 1.0
        sampling_ratio(int32, optional): number of sampling points in the interpolation grid. 
            If <=0, then grid points are adaptive to roi_width and pooled_w, likewise for height. Default: -1
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor:

        Output: The output of ROIAlignOp is a 4-D tensor with shape (num_rois, channels, pooled_h, pooled_w). The data type is float32 or float64.


    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()

            x = paddle.static.data(
                name='data', shape=[None, 256, 32, 32], dtype='float32')
            rois = paddle.static.data(
                name='rois', shape=[None, 4], dtype='float32')
            rois_num = paddle.static.data(name='rois_num', shape=[None], dtype='int32')
            align_out = ops.roi_align(input=x,
                                               rois=rois,
                                               ouput_size=(7, 7),
                                               spatial_scale=0.5,
                                               sampling_ratio=-1,
                                               rois_num=rois_num)
    """
    check_type(output_size, 'output_size', (int, tuple), 'roi_align')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        align_out = core.ops.roi_align(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale,
            "sampling_ratio", sampling_ratio, "aligned", aligned)
        return align_out


class RoIAlign(object):
    """
    RoI Align module

    For more details, please refer to the document of roi_align in
    in ppdet/modeing/ops.py

    Args:
        resolution (int): The output size, default 14
        spatial_scale (float): Multiplicative spatial scale factor to translate
            ROI coords from their input scale to the scale used when pooling.
            default 0.0625
        sampling_ratio (int): The number of sampling points in the interpolation
            grid, default 0
        canconical_level (int): The referring level of FPN layer with 
            specified level. default 4
        canonical_size (int): The referring scale of FPN layer with 
            specified scale. default 224
        start_level (int): The start level of FPN layer to extract RoI feature,
            default 0
        end_level (int): The end level of FPN layer to extract RoI feature,
            default 3
        aligned (bool): Whether to add offset to rois' coord in roi_align.
            default false
    """

    def __init__(self,
                 resolution=14,
                 spatial_scale=0.0625,
                 sampling_ratio=0,
                 canconical_level=4,
                 canonical_size=224,
                 start_level=0,
                 end_level=3,
                 aligned=False):
        super(RoIAlign, self).__init__()
        self.resolution = resolution
        self.spatial_scale = _to_list(spatial_scale)
        self.sampling_ratio = sampling_ratio
        self.canconical_level = canconical_level
        self.canonical_size = canonical_size
        self.start_level = start_level
        self.end_level = end_level
        self.aligned = aligned

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'spatial_scale': [1. / i.stride for i in input_shape]}

    def __call__(self, feats, roi, rois_num):
        roi = paddle.concat(roi) if len(roi) > 1 else roi[0]
        if len(feats) == 1:
            rois_feat = roi_align(
                feats[self.start_level],
                roi,
                self.resolution,
                self.spatial_scale[0],
                rois_num=rois_num,
                aligned=self.aligned)
        else:
            offset = 2
            k_min = self.start_level + offset
            k_max = self.end_level + offset
            rois_dist, restore_index, rois_num_dist = distribute_fpn_proposals(
                roi,
                k_min,
                k_max,
                self.canconical_level,
                self.canonical_size,
                rois_num=rois_num)
            rois_feat_list = []
            for lvl in range(self.start_level, self.end_level + 1):
                roi_feat = roi_align(
                    feats[lvl],
                    rois_dist[lvl],
                    self.resolution,
                    self.spatial_scale[lvl],
                    sampling_ratio=self.sampling_ratio,
                    rois_num=rois_num_dist[lvl],
                    aligned=self.aligned)
                rois_feat_list.append(roi_feat)
            rois_feat_shuffle = paddle.concat(rois_feat_list)
            rois_feat = paddle.gather(rois_feat_shuffle, restore_index)

        return rois_feat


class TwoFCHead(nn.Layer):
    """
    RCNN bbox head with Two fc layers to extract feature

    Args:
        in_channel (int): Input channel which can be derived by from_config
        out_channel (int): Output channel
        resolution (int): Resolution of input feature map, default 7
    """

    def __init__(self, in_channel=256, out_channel=1024, resolution=7):
        super(TwoFCHead, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        fan = in_channel * resolution * resolution
        self.fc6 = nn.Linear(
            in_channel * resolution * resolution,
            out_channel,
            weight_attr=paddle.ParamAttr(
                initializer=XavierUniform(fan_out=fan)))

        self.fc7 = nn.Linear(
            out_channel,
            out_channel,
            weight_attr=paddle.ParamAttr(initializer=XavierUniform()))

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_channel': s.channels}

    def forward(self, rois_feat):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6(rois_feat)
        fc6 = F.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = F.relu(fc7)
        return fc7


class BBoxHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner', 'bbox_loss']
    """
    RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        with_pool (bool): Whether to use pooling for the RoI feature.
        num_classes (int): The number of classes
        bbox_weight (List[float]): The weight to get the decode box 
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.],
                 bbox_loss=None):
        super(BBoxHead, self).__init__()
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner

        self.with_pool = with_pool
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.bbox_loss = bbox_loss

        self.bbox_score = nn.Linear(
            in_channel,
            self.num_classes + 1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)))

        self.bbox_delta = nn.Linear(
            in_channel,
            4 * self.num_classes,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.001)))

        self.assigned_label = None
        self.assigned_rois = None


    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if self.with_pool:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(scores, deltas, targets, rois,
                                 self.bbox_weight)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, rois_feat, self.head

    def get_loss(self, scores, deltas, targets, rois, bbox_weight):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}

        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets

        # bbox cls
        tgt_labels = paddle.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        valid_inds = paddle.nonzero(tgt_labels >= 0).flatten()
        if valid_inds.shape[0] == 0:
            loss_bbox[cls_name] = paddle.zeros([1], dtype='float32')
        else:
            tgt_labels = tgt_labels.cast('int64')
            tgt_labels.stop_gradient = True
            loss_bbox_cls = F.cross_entropy(
                input=scores, label=tgt_labels, reduction='mean')
            loss_bbox[cls_name] = loss_bbox_cls

        # bbox reg

        cls_agnostic_bbox_reg = deltas.shape[1] == 4

        fg_inds = paddle.nonzero(
            paddle.logical_and(tgt_labels >= 0, tgt_labels <
                               self.num_classes)).flatten()

        if fg_inds.numel() == 0:
            loss_bbox[reg_name] = paddle.zeros([1], dtype='float32')
            return loss_bbox

        if cls_agnostic_bbox_reg:
            reg_delta = paddle.gather(deltas, fg_inds)
        else:
            fg_gt_classes = paddle.gather(tgt_labels, fg_inds)

            reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = paddle.gather(deltas, fg_inds)
            reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])
        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True

        if self.bbox_loss is not None:
            reg_delta = self.bbox_transform(reg_delta)
            reg_target = self.bbox_transform(reg_target)
            loss_bbox_reg = self.bbox_loss(
                reg_delta, reg_target).sum() / tgt_labels.shape[0]
            loss_bbox_reg *= self.num_classes
        else:
            loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum(
            ) / tgt_labels.shape[0]

        loss_bbox[reg_name] = loss_bbox_reg

        return loss_bbox

    def bbox_transform(self, deltas, weights=[0.1, 0.1, 0.2, 0.2]):
        wx, wy, ww, wh = weights

        deltas = paddle.reshape(deltas, shape=(0, -1, 4))

        dx = paddle.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        dy = paddle.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        dw = paddle.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        dh = paddle.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dw = paddle.clip(dw, -1.e10, np.log(1000. / 16))
        dh = paddle.clip(dh, -1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = paddle.exp(dw)
        pred_h = paddle.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = paddle.reshape(x1, shape=(-1, ))
        y1 = paddle.reshape(y1, shape=(-1, ))
        x2 = paddle.reshape(x2, shape=(-1, ))
        y2 = paddle.reshape(y2, shape=(-1, ))

        return paddle.concat([x1, y1, x2, y2])

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois