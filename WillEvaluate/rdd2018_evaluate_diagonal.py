"""
时间：20210426
说明：用于道路损伤的目标检测模型的基于对角线的评价标准
"""
import json
import os

"""
用于将模型输出的数字标签转换成对应的字符标签
"""
number2label = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44']


def type_dict():
    return {
        'D00': 0,
        'D01': 0,
        'D10': 0,
        'D11': 0,
        'D20': 0,
        'D40': 0,
        'D43': 0,
        'D44': 0
    }


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


def diagonal_box_length(pre_box, match_box):
    # 对角线被匹配框包含的长度
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


def diagonal_evaluate(pre_box_queue, pre_label_queue, tar_box_queue, tar_label_queue):
    pre_nums = len(pre_box_queue)
    tar_nums = len(tar_box_queue)

    # 预测和GT匹配正确数量
    pre_match = type_dict()

    tar_match = type_dict()

    # 预测和GT标注数量
    detection_statistics = type_dict()

    gt_statistics = type_dict()

    # 预测框每个框的匹配分数
    pre_scores = []
    for i in range(pre_nums):
        pre_scores.append(0)

    # GT框每个框的匹配分数
    tar_scores = []
    for i in range(tar_nums):
        tar_scores.append(0)

    for label in pre_label_queue:
        detection_statistics[str(label)] += 1
    for label in tar_label_queue:
        if label == 'D30':  # 被抛弃的一个类
            continue
        gt_statistics[str(label)] += 1

    for i in range(pre_nums):
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
            match_diagonal_1 = diagonal_box_length(pre_box, match_box)
            # 副对角线
            pre_box2 = [pre_box[2], pre_box[1], pre_box[0], pre_box[3]]
            match_diagonal_2 = diagonal_box_length(pre_box2, match_box)
            # 选取匹配到长度较长的对角线作为作为匹配结果，以0.5作为阈值（参考来自己IOU匹配）
            tmp_score = (max(match_diagonal_1, match_diagonal_2) / pre_diagonal) / 0.5
            # 迭代进行，一个大框可能需要多个小框去匹配
            pre_scores[i] = pre_scores[i] + tmp_score

            # 2 用来计算GT框
            # 主对角线
            match_diagonal_1 = diagonal_box_length(tar_box, match_box)
            # 副对角线
            tar_box2 = [tar_box[2], tar_box[1], tar_box[0], tar_box[3]]
            match_diagonal_2 = diagonal_box_length(tar_box2, match_box)
            # 选取匹配到长度较长的对角线作为作为匹配结果，以0.5作为阈值（参考来自己IOU匹配）
            tmp_score = (max(match_diagonal_1, match_diagonal_2) / tar_diagonal) / 0.5
            # 迭代进行，一个大框可能需要多个小框去匹配
            tar_scores[j] = tar_scores[j] + tmp_score

    for i in range(pre_nums):
        if pre_scores[i] >= 1:
            pre_match[str(pre_label_queue[i])] += 1

    for i in range(tar_nums):
        if tar_scores[i] >= 1:
            tar_match[str(tar_label_queue[i])] += 1

    return pre_match, tar_match, detection_statistics, gt_statistics


def evaluate():
    prediction_json = "/home/WillEnglishPaper/mmdetection/RDD2018TestJson"
    target_json = "/home/WillEnglishPaper/mmdetection/RDD2018ValJson"

    prediction_list = os.listdir(prediction_json)
    target_list = os.listdir(target_json)
    prediction_list.sort()
    target_list.sort()

    match_prediction = type_dict()
    prediction_sta = type_dict()
    match_target = type_dict()
    target_sta = type_dict()
    precision = type_dict()
    recall = type_dict()
    F1_score = type_dict()
    F2_score = type_dict()

    assert len(prediction_list) == len(target_list), "预测数量和GT数量不一致"

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

        target_box_queue = target['bbox']
        target_label_queue = target['labels']

        match_pre, match_gt, pre_statistics, gt_statistics = diagonal_evaluate(prediction_box_queue,
                                                                               prediction_label_queue,
                                                                               target_box_queue,
                                                                               target_label_queue)
        # print(prediction_name)
        # print(match_pre)
        # print(match_gt)

        per_match_prediction = 0
        per_prediction_sta = 0
        per_match_target = 0
        per_target_sta = 0
        for key in match_prediction:
            per_match_prediction += match_pre[key]
            per_prediction_sta += pre_statistics[key]
            per_match_target += match_gt[key]
            per_target_sta += gt_statistics[key]
        #
        # per_pre = round(per_match_prediction / (per_prediction_sta + 1e-7), 4)
        # per_recall = round(per_match_target / per_target_sta, 4)
        # per_f1 = round(2 * (per_pre * per_recall) / ((per_pre + per_recall) + 1e-7), 4)
        # print(prediction_name)
        # print(per_pre)
        # print(per_recall)
        # print(per_f1)

        for key in match_prediction:
            match_prediction[key] += match_pre[key]
            prediction_sta[key] += pre_statistics[key]
            match_target[key] += match_gt[key]
            target_sta[key] += gt_statistics[key]

    for key in match_prediction:
        precision[key] = round(match_prediction[key] / (prediction_sta[key] + 1e-7), 4)
        recall[key] = round(match_target[key] / target_sta[key], 4)
        F1_score[key] = round(2 * (precision[key] * recall[key]) / ((precision[key] + recall[key]) + 1e-7), 4)
        F2_score[key] = round(5 * (precision[key] * recall[key]) / ((4 * precision[key] + recall[key]) + 1e-7), 4)

    F1_sum = 0
    F2_sum = 0
    for key in F1_score:
        F1_sum += F1_score[key]
        F2_sum += F2_score[key]
    res = {"precision": precision,
           "recall": recall,
           "F1": F1_score,
           "F2": F2_score,
           "AVG_F1": '%.4f' % (F1_sum / len(F1_score)),
           "AVG_F2": '%.4f' % (F2_sum / len(F2_score))}

    print(res["AVG_F1"])

    return res


if __name__ == "__main__":
    print(evaluate())
