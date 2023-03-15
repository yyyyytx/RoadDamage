"""
时间：20210426
说明：用于道路损伤的目标检测模型的基于对角线的评价标准
"""
import json
import os
import numpy as np

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


# 对角线被交集框覆盖的长度
def covered_diagonal_length(pre_box, match_box):
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


def diagonal_evaluate(pre_box_queue, pre_label_queue, pre_socres_queue, tar_box_queue, tar_label_queue, pre_start):
    pre_nums = len(pre_box_queue)
    tar_nums = len(tar_box_queue)

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

    # 用来记录某个预测框使得哪些gt框命中
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

        # 第四列表示某个预测边界框索引位置
        # 由于是所有图片一起统计，因此加上了偏移量pre_start
        pre[i][3] = i + pre_start

    return pre, pre_match_gt


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

    pre_start = 0
    pre_all = np.zeros((0, 4))
    pre_match_gt_all = []
    tar_label_queue_all = []
    # 一张图片一张图片处理
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
        for cls_id in range(len(prediction['labels'])):
            prediction_label_queue.append(number2label[prediction['labels'][cls_id]])
        prediction_score_queue = prediction['scores']

        target_box_queue = []
        target_label_queue = []
        for k in range(len(target['shapes'])):
            target_box_queue.append(target['shapes'][k]['points'][0] + target['shapes'][k]['points'][1])
            target_label_queue.append(target['shapes'][k]['label'])
        tar_label_queue_all.extend(target_label_queue)

        pre_img, pre_match_gt = diagonal_evaluate(prediction_box_queue, prediction_label_queue,
                                                  prediction_score_queue,
                                                  target_box_queue,
                                                  target_label_queue,
                                                  pre_start)

        pre_all = np.concatenate((pre_all, pre_img), axis=0)
        pre_match_gt_all.extend(pre_match_gt)
        # TODO 输出每张图片的ap
        # print(prediction_name)
        # print(ap_per_img)
        pre_start += len(prediction_box_queue)

    all_cls_ap = 0
    cls_count = 0
    # 排序 保证置信度由高到底
    pre_all = pre_all[pre_all[:, 0].argsort()]
    pre_all = pre_all[::-1]
    # 按类别统计ap
    for cls_id in range(len(number2label)):

        indx = pre_all[:, 1] == cls_id
        pre_cls_id = pre_all[indx]

        # 用以统计tp和gt命中数量
        tp_count = 0
        gt_count = 0

        precisions = []
        recalls = []

        # 某种类别下所有的gt数
        cls_gt_nums = tar_label_queue_all.count(number2label[cls_id])

        # 既没有该类别的gt也没有该类别的预测
        if cls_gt_nums == 0 and len(pre_cls_id) == 0:
            continue

        cls_count += 1
        for j in range(len(pre_cls_id)):
            # 如果该预测是TP 第三列表示是否为tp
            if pre_cls_id[j][2] == 1:
                tp_count += 1
                # TODO 这里做了妥协 因为pre和GT是多对多的关系 非传统的一对一
                # gt命中匹配是依据pre预测框匹配的数量
                gt_count += len(pre_match_gt_all[int(pre_cls_id[j][3])])

            p_t = tp_count / (j + 1)
            r_t = gt_count / (cls_gt_nums + 1e-7)
            precisions.append(p_t)
            recalls.append(r_t)

        cls_ap = average_precision(np.array(recalls), np.array(precisions), mode='area')

        all_cls_ap += cls_ap

    mAP = all_cls_ap / cls_count

    print("area_mAP：{:.3f}".format(mAP))

    return mAP


if __name__ == "__main__":
    print(liqing_evaluate())
