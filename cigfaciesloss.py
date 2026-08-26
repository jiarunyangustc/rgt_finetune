import numpy as np
import torch
import torch.nn as nn


def compute_CIGLoss(input, indexs, ciglabel_dir=None, seeds_num=500, loss_type="L1"):
    if ciglabel_dir is None:
        raise ValueError("ciglabel_dir is not defined")

    bs, c, h, w = input.size()
    sum_loss = input.sum() * 0.0

    for bi in range(bs):
        input_bs = input[bi, 0, :, :]
        input_bs = ((input_bs - torch.mean(input_bs)) / torch.max(input_bs)) * 10
        paths_all = np.load(ciglabel_dir + str(indexs[bi].tolist()) + ".npy", allow_pickle=True).item()
        pn = len(paths_all)

        for pi in range(pn):
            paths_pixels = torch.tensor(paths_all[str(pi)], dtype=torch.int32, device=input.device)
            path_pixel_num = len(paths_pixels)
            if path_pixel_num == 0:
                continue

            local_input = input_bs[paths_pixels[:, 0], paths_pixels[:, 1]]
            path_mean_val = torch.mean(local_input)
            diff = local_input - path_mean_val.detach()

            if loss_type == "L1":
                local_loss = torch.sum(torch.abs(diff)) / path_pixel_num
            elif loss_type == "L2":
                local_loss = torch.sum(diff ** 2) / path_pixel_num
            else:
                raise ValueError("Invalid CIG loss type: choose L1 or L2")
            sum_loss = sum_loss + local_loss

    return sum_loss / bs


def _segment_flatten_loss_one(pred, seg, loss_type, beta, min_points,
                              equal_segment_weight, detach_center,
                              depth_weight_gamma=0.0, qual=None):
    pred_flat = pred.reshape(-1)
    seg_flat = seg.reshape(-1).long()
    valid_mask = seg_flat > 0
    if not valid_mask.any():
        return pred.sum() * 0.0, False

    pred_valid = pred_flat[valid_mask]
    seg_valid = seg_flat[valid_mask]

    # 深度加权：dim0 视为深度轴，segment 平均深度越大权重越高
    # w = 1 + gamma * (depth/H)，gamma=0 时完全退化为原行为
    depth_valid = None
    if depth_weight_gamma > 0:
        H = pred.shape[0]
        zgrid = torch.arange(H, device=pred.device, dtype=pred.dtype)
        depth_flat = zgrid.view(-1, *([1] * (pred.dim() - 1))).expand_as(pred).reshape(-1)
        depth_valid = depth_flat[valid_mask]
    # 质量加权：qual 为逐像素权重图（与 pred 同形），段内取均值作为该段权重，
    # 归一化加权（/sum w）——低权段以折扣参与，不稀释高权段的约束。qual=None 时退化为原行为
    qual_valid = None
    if qual is not None:
        qual_valid = qual.reshape(-1)[valid_mask]
    unique_segs = torch.unique(seg_valid)
    if unique_segs.numel() < 1:
        return pred.sum() * 0.0, False

    max_label = int(seg_valid.max().item()) + 1
    ones = torch.ones_like(pred_valid)
    count_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    count_per_seg.scatter_add_(0, seg_valid, ones)

    valid_seg_mask = count_per_seg[unique_segs] >= float(min_points)
    if not torch.any(valid_seg_mask):
        return pred.sum() * 0.0, False

    keep_seg_ids = unique_segs[valid_seg_mask]
    keep_lut = torch.zeros(max_label, device=pred.device, dtype=torch.bool)
    keep_lut[keep_seg_ids] = True
    point_keep = keep_lut[seg_valid]
    pred_valid = pred_valid[point_keep]
    seg_valid = seg_valid[point_keep]
    if depth_valid is not None:
        depth_valid = depth_valid[point_keep]
    if qual_valid is not None:
        qual_valid = qual_valid[point_keep]
    if pred_valid.numel() == 0:
        return pred.sum() * 0.0, False

    sum_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    sum_per_seg.scatter_add_(0, seg_valid, pred_valid)
    count_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    count_per_seg.scatter_add_(0, seg_valid, torch.ones_like(pred_valid))
    center_per_seg = sum_per_seg / torch.clamp(count_per_seg, min=1.0)
    center_at_pixel = center_per_seg[seg_valid]
    if detach_center:
        center_at_pixel = center_at_pixel.detach()

    diff = pred_valid - center_at_pixel
    loss_name = loss_type.lower()
    if loss_name in ["l1", "mae"]:
        point_loss = diff.abs()
    elif loss_name in ["l2", "mse"]:
        point_loss = diff ** 2
    elif loss_name in ["smoothl1", "smooth_l1", "huber"]:
        abs_diff = diff.abs()
        if beta <= 0:
            point_loss = abs_diff
        else:
            point_loss = torch.where(
                abs_diff < beta,
                0.5 * diff ** 2 / beta,
                abs_diff - 0.5 * beta,
            )
    else:
        raise ValueError("Invalid segment loss type: choose L1, L2, or SmoothL1")

    if equal_segment_weight:
        sum_loss_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
        sum_loss_per_seg.scatter_add_(0, seg_valid, point_loss)
        loss_per_seg = sum_loss_per_seg / torch.clamp(count_per_seg, min=1.0)
        uniq = torch.unique(seg_valid)
        w = None
        if depth_valid is not None:
            sum_depth_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
            sum_depth_per_seg.scatter_add_(0, seg_valid, depth_valid)
            depth_per_seg = sum_depth_per_seg / torch.clamp(count_per_seg, min=1.0)
            w = 1.0 + depth_weight_gamma * depth_per_seg / max(pred.shape[0] - 1, 1)
        if qual_valid is not None:
            sum_qual_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
            sum_qual_per_seg.scatter_add_(0, seg_valid, qual_valid)
            qual_per_seg = sum_qual_per_seg / torch.clamp(count_per_seg, min=1.0)
            w = qual_per_seg if w is None else w * qual_per_seg
        if w is not None:
            wu = torch.clamp(w[uniq], min=0.0)
            if wu.sum() <= 0:
                return pred.sum() * 0.0, False
            loss = (loss_per_seg[uniq] * wu).sum() / wu.sum()
        else:
            loss = loss_per_seg[uniq].mean()
    else:
        if qual_valid is not None:
            qsum = torch.clamp(qual_valid, min=0.0).sum()
            if qsum <= 0:
                return pred.sum() * 0.0, False
            loss = (point_loss * torch.clamp(qual_valid, min=0.0)).sum() / qsum
        else:
            loss = point_loss.mean()

    if torch.isnan(loss) or torch.isinf(loss):
        return pred.sum() * 0.0, False
    return loss, True


def compute_SegmentLoss(input, segment, loss_type="SmoothL1", beta=0.01,
                        min_points=2, equal_segment_weight=True,
                        detach_center=True, depth_weight_gamma=0.0,
                        weight=None, cross_slice=False):
    """
    Self-flatten segment loss without teacher labels.

    The target for each segment is the current prediction's own segment-wise
    mean. The target is detached by default, so this loss only flattens RGT
    within each segment and does not introduce an absolute teacher RGT value.

    input:   [B, 1, ...] predicted RGT
    segment: [B, 1, ...] integer segment labels, <=0 ignored
    weight:  optional [B, 1, ...] per-pixel quality weight (segment-wise mean
             is used, normalized); None keeps original behavior
    """
    assert input.shape[:2] == segment.shape[:2] and input.shape[1] == 1, \
        "Expect input and segment shapes [B,1,...]"

    if cross_slice:
        # 跨切片分组: 整个 batch 一次 flatten, 全局 segment ID 使同一物理段
        # 在相邻切片间共用同一个中心 —— 提供跨片积分偏置耦合。
        # 前提: batch 为相邻切片(shuffle=False)且 segment ID 跨切片全局一致。
        # permute 把深度轴放到 dim0, 保持 depth_weight_gamma 的 zgrid 语义正确。
        loss, valid = _segment_flatten_loss_one(
            input[:, 0].permute(1, 0, 2), segment[:, 0].permute(1, 0, 2),
            loss_type, beta, min_points, equal_segment_weight, detach_center,
            depth_weight_gamma,
            qual=None if weight is None else weight[:, 0].permute(1, 0, 2),
        )
        if not valid:
            return input.sum() * 0.0
        return loss

    total = input.sum() * 0.0
    valid_batches = 0
    for bi in range(input.shape[0]):
        loss, valid = _segment_flatten_loss_one(
            input[bi, 0], segment[bi, 0], loss_type, beta, min_points,
            equal_segment_weight, detach_center, depth_weight_gamma,
            qual=None if weight is None else weight[bi, 0],
        )
        if valid:
            total = total + loss
            valid_batches += 1

    if valid_batches == 0:
        return input.sum() * 0.0
    return total / valid_batches


class SegmentLoss(nn.Module):
    """Segment flatten loss without teacher RGT labels."""

    def __init__(self, loss_type="SmoothL1", beta=0.01, min_points=2,
                 equal_segment_weight=True, detach_center=True,
                 depth_weight_gamma=0.0, cross_slice=False):
        super(SegmentLoss, self).__init__()
        self.name = 'SegmentLoss'
        self.loss_type = loss_type
        self.beta = beta
        self.min_points = min_points
        self.equal_segment_weight = equal_segment_weight
        self.detach_center = detach_center
        self.depth_weight_gamma = depth_weight_gamma
        self.cross_slice = cross_slice

    def forward(self, input, segment, weight=None):
        return compute_SegmentLoss(
            input, segment,
            loss_type=self.loss_type,
            beta=self.beta,
            min_points=self.min_points,
            equal_segment_weight=self.equal_segment_weight,
            detach_center=self.detach_center,
            depth_weight_gamma=self.depth_weight_gamma,
            weight=weight,
            cross_slice=self.cross_slice,
        )


class SegmentLoss_2d(SegmentLoss):
    pass


def _segment_order_loss_one(pred, seg, min_points=5, min_depth_gap=4.0,
                            margin=0.02, max_segments=128):
    pred_flat = pred.reshape(-1)
    seg_flat = seg.reshape(-1).long()
    valid_mask = seg_flat > 0
    if not valid_mask.any():
        return pred.sum() * 0.0, False

    pred_valid = pred_flat[valid_mask]
    seg_valid = seg_flat[valid_mask]
    unique_segs = torch.unique(seg_valid)
    if unique_segs.numel() < 2:
        return pred.sum() * 0.0, False

    max_label = int(seg_valid.max().item()) + 1
    ones = torch.ones_like(pred_valid)
    count_per_seg = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    count_per_seg.scatter_add_(0, seg_valid, ones)

    keep = unique_segs[count_per_seg[unique_segs] >= float(min_points)]
    if keep.numel() < 2:
        return pred.sum() * 0.0, False

    if keep.numel() > max_segments:
        _, idx = torch.topk(count_per_seg[keep], k=max_segments)
        keep = keep[idx]

    keep_lut = torch.zeros(max_label, device=pred.device, dtype=torch.bool)
    keep_lut[keep] = True
    point_keep = keep_lut[seg_valid]
    pred_valid = pred_valid[point_keep]
    seg_valid = seg_valid[point_keep]
    if pred_valid.numel() == 0:
        return pred.sum() * 0.0, False

    zgrid = torch.arange(pred.shape[0], device=pred.device, dtype=pred.dtype)
    depth_flat = zgrid.view(-1, *([1] * (pred.dim() - 1))).expand_as(pred).reshape(-1)
    depth_valid = depth_flat[valid_mask][point_keep]

    sum_pred = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    sum_depth = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    count = torch.zeros(max_label, device=pred.device, dtype=pred.dtype)
    sum_pred.scatter_add_(0, seg_valid, pred_valid)
    sum_depth.scatter_add_(0, seg_valid, depth_valid)
    count.scatter_add_(0, seg_valid, torch.ones_like(pred_valid))

    mean_pred = sum_pred[keep] / torch.clamp(count[keep], min=1.0)
    mean_depth = sum_depth[keep] / torch.clamp(count[keep], min=1.0)
    order = torch.argsort(mean_depth)
    mean_pred = mean_pred[order]
    mean_depth = mean_depth[order]

    depth_i = mean_depth[:, None]
    depth_j = mean_depth[None, :]
    pred_i = mean_pred[:, None]
    pred_j = mean_pred[None, :]
    pair_mask = depth_j > depth_i + float(min_depth_gap)
    if not pair_mask.any():
        return pred.sum() * 0.0, False

    violation = torch.relu(float(margin) - (pred_j - pred_i))
    loss = violation[pair_mask].mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return pred.sum() * 0.0, False
    return loss, True


def compute_SegmentOrderLoss(input, segment, min_points=5, min_depth_gap=4.0,
                             margin=0.02, max_segments=128):
    """
    Weak relative-order loss for segments without teacher RGT labels.

    It sorts segments by geometric mean depth and penalizes cases where deeper
    segments have smaller mean predicted RGT than shallower segments.
    """
    assert input.shape[:2] == segment.shape[:2] and input.shape[1] == 1, \
        "Expect input and segment shapes [B,1,...]"

    total = input.sum() * 0.0
    valid_batches = 0
    for bi in range(input.shape[0]):
        loss, valid = _segment_order_loss_one(
            input[bi, 0], segment[bi, 0],
            min_points=min_points,
            min_depth_gap=min_depth_gap,
            margin=margin,
            max_segments=max_segments,
        )
        if valid:
            total = total + loss
            valid_batches += 1

    if valid_batches == 0:
        return input.sum() * 0.0
    return total / valid_batches


class SegmentOrderLoss(nn.Module):
    """Relative depth-order loss between segments."""

    def __init__(self, min_points=5, min_depth_gap=4.0,
                 margin=0.02, max_segments=128):
        super(SegmentOrderLoss, self).__init__()
        self.name = 'SegmentOrderLoss'
        self.min_points = min_points
        self.min_depth_gap = min_depth_gap
        self.margin = margin
        self.max_segments = max_segments

    def forward(self, input, segment):
        return compute_SegmentOrderLoss(
            input, segment,
            min_points=self.min_points,
            min_depth_gap=self.min_depth_gap,
            margin=self.margin,
            max_segments=self.max_segments,
        )


class CIGLoss(nn.Module):
    def __init__(self, seeds_num=500, loss_type="L1", ciglabel_dir=None):
        super(CIGLoss, self).__init__()
        self.name = 'CIGLoss'
        self.loss_type = loss_type
        self.ciglabel_dir = ciglabel_dir
        self.seeds_num = seeds_num

    def forward(self, input, indexs):
        return compute_CIGLoss(
            input, indexs,
            ciglabel_dir=self.ciglabel_dir,
            seeds_num=self.seeds_num,
            loss_type=self.loss_type,
        )


def calculate_normal_loss(input, normal, linearity, eps=1e-20):
    bs, _, h, w = input.size()
    gx = torch.zeros_like(normal)
    grads = torch.gradient(input, dim=(2, 3))
    gx[:, :, :, :, 0] = grads[0]
    gx[:, :, :, :, 1] = grads[1]
    cosloss = nn.CosineSimilarity(dim=4, eps=eps)
    loss = 1 - cosloss(gx, normal)
    return torch.sum(loss) / bs / h / w


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, input, normal, linearity):
        return calculate_normal_loss(input, normal, linearity)
