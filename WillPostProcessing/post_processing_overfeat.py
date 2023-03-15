#

import numpy as np
import torch


def boxlist_iou(box):
    """
    计算最终预测边界框自己和自己的IOU
    边界框编码顺序需要是(xmin, ymin, xmax, ymax).

    参数:
      box: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) iou, sized [N,N].
      (tensor) inter, sized [N,N].
      (tensor) area, sized [N].

    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    lt = torch.max(box[:, None, :2], box[:, :2])  # [N,N,2]
    rb = torch.min(box[:, None, 2:], box[:, 2:])  # [N,N,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,N,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,N]

    iou = inter / (area[:, None] + area - inter)
    return iou.clamp(max=1.0), inter, area


def post_connect_overfeat(bboxes, labels):
    # TODO 参考OverFeat 可用以替换自己的边界框合并后处理

    while True:

        boxes = bboxes[:, :4]

        overlaps, _, _ = boxlist_iou(torch.from_numpy(boxes))

        # IOU矩阵对角线上全为1.0 不考虑这个
        self_mask = overlaps < 1.0
        overlaps *= self_mask

        # 我们仅合并类别一样的边界框
        label_x = labels.reshape((len(boxes), 1))
        label_y = labels.reshape((1, len(boxes)))
        label_mask = label_x == label_y

        overlaps *= torch.from_numpy(label_mask)

        if len(overlaps) == 0 or overlaps is None or overlaps.max() < 0.25:
            return bboxes, labels

        index = torch.argmax(overlaps)

        # 求出iou矩阵中最大值的坐标
        if (index + 1) % len(boxes) == 0:
            i = (index + 1) // len(boxes) - 1
            j = len(boxes) - 1
        else:
            i = (index + 1) // len(boxes)
            j = ((index + 1) % len(boxes)) - 1

        # 将 i，j合并
        bboxes[j][0] = min(bboxes[i][0], bboxes[j][0])
        bboxes[j][1] = min(bboxes[i][1], bboxes[j][1])
        bboxes[j][2] = max(bboxes[i][2], bboxes[j][2])
        bboxes[j][3] = max(bboxes[i][3], bboxes[j][3])
        bboxes[j][-1] = max(bboxes[i][-1], bboxes[j][-1])

        # 合并的新边界框已经覆盖到j，因此将i删除
        bboxes = np.delete(bboxes, i, axis=0)
        labels = np.delete(labels, i, axis=0)
