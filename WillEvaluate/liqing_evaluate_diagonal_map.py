"""
时间：20210426
说明：用于道路损伤的目标检测模型的基于对角线的评价标准
"""
import json
import os
import numpy as np

from WillEvaluate.voc_map import voc_ap
from mmdet.core import average_precision

"""
用于将模型输出的数字标签转换成对应的字符标签
"""
number2label = [
    "ZXLF",
    "HXLF",
    "GFXB",
    "KZXB",
    "KC",
    "KZLF",
]


# 计算两个边界框的IOU
def box_iou(xtl_1, ytl_1, xbr_1, ybr_1, xtl_2, ytl_2, xbr_2, ybr_2):
    xtl = max(xtl_1, xtl_2)
    xbr = min(xbr_1, xbr_2)
    ytl = max(ytl_1, ytl_2)
    ybr = min(ybr_1, ybr_2)

    w = max(0., xbr - xtl)
    h = max(0., ybr - ytl)

    intersection = w * h
    union = (xbr_1 - xtl_1) * (ybr_1 - ytl_1) + (xbr_2 - xtl_2) * (ybr_2 - ytl_2) - intersection

    return intersection, intersection / union


# 求两条线段的交点
def segments_intersection(ax, ay, bx, by, cx, cy, dx, dy):
    # 三角形abc 面积的2倍
    area_abc = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)

    # 三角形abd 面积的2倍
    area_abd = (ax - dx) * (by - dy) - (ay - dy) * (bx - dx)

    # 面积符号相同则两点在线段同侧,不相交 (对点在线段上的情况,本例当作不相交处理);
    if area_abc * area_abd > 0:
        return -1, -1

    # 三角形cda 面积的2倍
    area_cda = (cx - ax) * (dy - ay) - (cy - ay) * (dx - ax)

    # 三角形cdb 面积的2倍
    # 注意: 这里有一个小优化.不需要再用公式计算面积,而是通过已知的三个面积加减得出.
    area_cdb = area_cda + area_abc - area_abd
    if area_cda * area_cdb > 0:
        return -1, -1

    # 计算交点坐标
    t = area_cda / (area_abd - area_abc)
    deltax = t * (bx - ax)
    deltay = t * (by - ay)
    return round(ax + deltax, 6), round(ay + deltay, 6)


def covered_diagonal_length(pre_box, match_box):
    # 对角线被交集框覆盖的长度
    # 分别求pre_box的对角线与match_box的四条边的交点
    set_points = set()

    x1, y1 = segments_intersection(pre_box[0], pre_box[1], pre_box[2], pre_box[3], match_box[0], match_box[1],
                                   match_box[2],
                                   match_box[1])
    if x1 != -1:
        set_points.add((x1, y1))

    x2, y2 = segments_intersection(pre_box[0], pre_box[1], pre_box[2], pre_box[3], match_box[2], match_box[1],
                                   match_box[2],
                                   match_box[3])
    if x2 != -1:
        set_points.add((x2, y2))

    x3, y3 = segments_intersection(pre_box[0], pre_box[1], pre_box[2], pre_box[3], match_box[2], match_box[3],
                                   match_box[0],
                                   match_box[3])
    if x3 != -1:
        set_points.add((x3, y3))

    x4, y4 = segments_intersection(pre_box[0], pre_box[1], pre_box[2], pre_box[3], match_box[0], match_box[3],
                                   match_box[0],
                                   match_box[1])
    if x4 != -1:
        set_points.add((x4, y4))

    # 一条线段与匹配矩形有且仅有两个交点(四个坐标)
    if len(set_points) != 2:
        return 0
    else:
        listPoints = list(set_points)
        return ((listPoints[1][0] - listPoints[0][0]) ** 2 + (listPoints[1][1] - listPoints[0][1]) ** 2) ** 0.5


def diagonal_evaluate(pre_box_queue, pre_label_queue, pre_socres_queue, tar_box_queue, tar_label_queue):
    pre_nums = len(pre_box_queue)
    tar_nums = len(tar_box_queue)

    # 预测和GT标注数量
    detection_statistics = {
        "ZXLF": 0,
        "HXLF": 0,
        "GFXB": 0,
        "KZXB": 0,
        "KC": 0,
        "KZLF": 0,
    }
    gt_statistics = {
        "ZXLF": 0,
        "HXLF": 0,
        "GFXB": 0,
        "KZXB": 0,
        "KC": 0,
        "KZLF": 0,
    }

    pre = np.zeros((pre_nums, 4))

    # 预测框每个框的匹配分数
    pre_scores = []
    for i in range(pre_nums):
        pre_scores.append(0)
        # 第一列表示预测置信度
        pre[i][0] = pre_socres_queue[i]
        # 第二列表示类别
        pre[i][1] = number2label.index(pre_label_queue[i])

    # GT框每个框的匹配分数
    tar_scores = []
    for i in range(tar_nums):
        tar_scores.append(0)

    for label in pre_label_queue:
        detection_statistics[str(label)] += 1
    for label in tar_label_queue:
        gt_statistics[str(label)] += 1

    pre_match_gt = []
    for i in range(pre_nums):
        pre_match_gt.append([])
        pre_box = pre_box_queue[i]
        pre_label = pre_label_queue[i]
        pre_diagonal = ((pre_box[2] - pre_box[0]) ** 2 + (pre_box[3] - pre_box[1]) ** 2) ** 0.5

        for j in range(tar_nums):

            # 预测和GT类别不一致 或者GT分数大于等于1（代表该GT已经不能再去匹配）
            if pre_label != tar_label_queue[j] or tar_scores[j] >= 1:
                continue

            tar_box = tar_box_queue[j]
            tar_diagonal = ((tar_box[2] - tar_box[0]) ** 2 + (tar_box[3] - tar_box[1]) ** 2) ** 0.5

            _, IOU = box_iou(pre_box[0], pre_box[1], pre_box[2], pre_box[3], tar_box[0], tar_box[1], tar_box[2],
                             tar_box[3])
            # 两个框不相交
            if IOU == 0:
                continue

            # 以两个框的交集作为匹配框
            match_box = [max(pre_box[0], tar_box[0]), max(pre_box[1], tar_box[1]), min(pre_box[2], tar_box[2]),
                         min(pre_box[3], tar_box[3])]

            # 1 用来匹配预测框
            # 主对角线
            match_diagonal_1 = covered_diagonal_length(pre_box, match_box)
            # 副对角线
            pre_box2 = [pre_box[2], pre_box[1], pre_box[0], pre_box[3]]
            match_diagonal_2 = covered_diagonal_length(pre_box2, match_box)
            # 选取匹配到长度较长的对角线作为作为匹配结果，以0.5作为阈值（参考来自己IOU匹配）
            tmp_score = (max(match_diagonal_1, match_diagonal_2) / pre_diagonal) / 0.5
            # 迭代进行，一个大框可能需要多个小框去匹配
            pre_scores[i] = pre_scores[i] + tmp_score

            # 2 用来计算GT框
            # 主对角线
            match_diagonal_1 = covered_diagonal_length(tar_box, match_box)
            # 副对角线
            tar_box2 = [tar_box[2], tar_box[1], tar_box[0], tar_box[3]]
            match_diagonal_2 = covered_diagonal_length(tar_box2, match_box)
            # 选取匹配到长度较长的对角线作为作为匹配结果，以0.5作为阈值（参考来自己IOU匹配）
            tmp_score = (max(match_diagonal_1, match_diagonal_2) / tar_diagonal) / 0.5
            # 迭代进行，一个大框可能需要多个小框去匹配
            tar_scores[j] = tar_scores[j] + tmp_score

            if tar_scores[j] >= 1.0:
                pre_match_gt[i].append(j)

    for i in range(pre_nums):
        if pre_scores[i] >= 1:
            # 第三列表示是否为tp
            pre[i][2] = 1

        # 第四列表示索引
        pre[i][3] = i

    # 排序 保证置信度由高到底
    pre = pre[pre[:, 0].argsort()]
    pre = pre[::-1]

    all_cls_ap = 0
    cls_count = 0
    # 按类别统计ap
    for i in range(len(number2label)):
        indx = pre[:, 1] == i
        cls_i = pre[indx]
        precisions = []
        tp_count = 0
        gt_count = 0
        recalls = []
        cls_gt_nums = tar_label_queue.count(number2label[i])

        # 既没有该类别的gt也没有该类别的预测
        if cls_gt_nums == 0 and len(cls_i) == 0:
            continue
        cls_count += 1
        for j in range(len(cls_i)):
            # 如果该类别不存在GT 跳出
            # if cls_gt_nums == 0:
            #     break
            if cls_i[j][2] == 1:
                tp_count += 1
                gt_count += len(pre_match_gt[int(cls_i[j][3])])
            p_t = tp_count / (j + 1)
            r_t = gt_count / (cls_gt_nums + 1e-7)
            precisions.append(p_t)
            recalls.append(r_t)

        cls_ap = average_precision(np.array(recalls), np.array(precisions), mode='area')

        all_cls_ap += cls_ap

    ap_img = all_cls_ap / cls_count
    return ap_img


def liqing_evaluate():
    prediction_json = "/home/will/mmdetection/prediction_json/liqing"
    target_json = "/home/will/mmdetection/target_json/liqing"

    # prediction_json = "/home/will/mmdetection/prediction_json/tmp"
    # target_json = "/home/will/mmdetection/target_json/tmp"

    prediction_list = os.listdir(prediction_json)
    target_list = os.listdir(target_json)
    prediction_list.sort()
    target_list.sort()

    assert len(prediction_list) == len(target_list), "预测数量和GT数量不一致"

    all_img_ap = 0
    for prediction_name, target_name in zip(prediction_list, target_list):
        if prediction_name != target_name:
            print("预测json文件与标注json文件名对应不一致")

        prediction_path = os.path.join(prediction_json, prediction_name)
        with open(prediction_path, 'rb') as f1:
            prediction = json.load(f1)

        target_path = os.path.join(target_json, target_name)
        with open(target_path, 'rb') as f2:
            target = json.load(f2)

        prediction_box_queue = prediction['bbox']
        prediction_label_queue = []
        for i in range(len(prediction['labels'])):
            prediction_label_queue.append(number2label[prediction['labels'][i]])
        prediction_score_queue = prediction['scores']

        target_box_queue = []
        target_label_queue = []
        for k in range(len(target['shapes'])):
            target_box_queue.append(target['shapes'][k]['points'][0] + target['shapes'][k]['points'][1])
            target_label_queue.append(target['shapes'][k]['label'])

        ap_per_img = diagonal_evaluate(prediction_box_queue, prediction_label_queue,
                                       prediction_score_queue,
                                       target_box_queue,
                                       target_label_queue)

        # TODO 输出每张图片的ap
        # print(prediction_name)
        # print(ap_per_img)

        all_img_ap += ap_per_img

    mAP = all_img_ap / len(prediction_list)

    print("11points_mAP：{:.3f}".format(mAP))
    print("area_mAP：{:.3f}".format(mAP))

    return mAP


if __name__ == "__main__":
    print(liqing_evaluate())
