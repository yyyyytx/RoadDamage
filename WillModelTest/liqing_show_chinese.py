"""
时间：20210426
说明：重写了mmdet的检测结果显示文件主要是引入我们自己的后处理
"""

import cv2
import mmcv
import numpy as np
import torch
from PIL import ImageDraw, Image, ImageFont
from mmcv import color_val
from mmcv.image import imread, imwrite


def compute_colors_for_labels(labels):
    """
    根据类添加固定颜色的函数
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    labels = torch.tensor(labels).int()
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      text_color='blue',
                      thickness=1,
                      font_scale=0.5,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    # TODO 针对道路损伤检测 后处理主要包括下面四步
    # 阈值筛选
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    # 检测后处理(连接同类别的框)
    # bboxes, labels = post_connect(bboxes, labels)
    # 检测后处理（去掉重叠框）
    # bboxes, labels = post_newnms(bboxes, labels)
    # 检测后处理（去掉小框）
    # bboxes, labels = post_small(bboxes, labels)

    # 检测后处理（对角线抑制）TODO 大论文里面再用
    # bboxes, labels = post_diagonal(bboxes, labels)

    colors = compute_colors_for_labels(labels).tolist()
    text_color = color_val(text_color)
    img = np.ascontiguousarray(img)

    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("/home/will/mmdetection/WillModelTest/simhei.ttf", 50, encoding="utf-8")

    for bbox, label, color in zip(bboxes, labels, colors):
        bbox_int = bbox.astype(np.int32)

        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        # if len(bbox) > 4:
        #     label_text += f'|{bbox[-1]:.02f}'
        # TODO +5是为了不让文字和边界框重叠
        x, y = bbox_int[0] + 5, bbox_int[1] + 5

        draw.text((x, y), label_text, text_color, font=font)

        # cv2.putText(img, label_text, (x, y),
        #             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, 4)

    img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    for bbox, _, color in zip(bboxes, labels, colors):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, tuple(color), thickness=thickness)

    if out_file is not None:
        imwrite(img, out_file)
    return img, bboxes, labels


def show_result(
        img,
        detection_result,
        CLASSES,
        score_thr=0.3,
        text_color='blue',
        thickness=10,
        font_scale=3,
        out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): 需要展示的图片.
        detection_result (Tensor or tuple): 边界框等结果信息.
        CLASSES(list[str]): 每个类别的名字.
        score_thr (float, optional): 最小置信度阈值.
            Default: 0.3.
        bbox_color (str or tuple or :obj:`Color`): 边界框颜色.
        text_color (str or tuple or :obj:`Color`): 文本颜色.
        thickness (int): 边界框粗细.
        font_scale (float): 文本大小.
        out_file (str or None): 存储路径.
            Default: None.
    """
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(detection_result, tuple):
        bbox_result, segm_result = detection_result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = detection_result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = segms[i]
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    _, bboxes, labels = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=CLASSES,
        score_thr=score_thr,
        text_color=text_color,
        thickness=thickness,
        font_scale=font_scale,
        out_file=out_file)
    return bboxes, labels
