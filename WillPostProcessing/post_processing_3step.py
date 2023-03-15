def box_iou(xtl_1, ytl_1, xbr_1, ybr_1, xtl_2, ytl_2, xbr_2, ybr_2):
    xx = max(xtl_1, xtl_2)
    XX = min(xbr_1, xbr_2)
    yy = max(ytl_1, ytl_2)
    YY = min(ybr_1, ybr_2)
    m = max(0., XX - xx)
    n = max(0., YY - yy)
    intersection = m * n
    union = (xbr_1 - xtl_1) * (ybr_1 - ytl_1) + (xbr_2 - xtl_2) * (ybr_2 - ytl_2) - intersection
    return intersection, intersection / (union + 1e-7)


def post_connect(bboxes, labels):
    # TODO 可能有更好的合并方式，参考OverFeat
    boxes = bboxes[:, :4]
    keep = []
    length = len(boxes)
    if length == 0:
        return bboxes, labels
    for i in range(length):
        box1 = boxes[i]
        top_left1, bottom_right1 = box1[:2].tolist(), box1[2:].tolist()  # 左上角与右下角坐标
        xtl_1, ytl_1 = top_left1
        xbr_1, ybr_1 = bottom_right1
        flag = 1
        for j in range(i + 1, length):
            box2 = boxes[j]
            top_left2, bottom_right2 = box2[:2].tolist(), box2[2:].tolist()  # 左上角与右下角坐标
            xtl_2, ytl_2 = top_left2
            xbr_2, ybr_2 = bottom_right2

            _, IOU = box_iou(xtl_1, ytl_1, xbr_1, ybr_1, xtl_2, ytl_2, xbr_2, ybr_2)
            if IOU == 0:
                continue
            # 合并到j框，去除i框
            elif IOU > 0.25 and labels[i] == labels[j]:  # 求两个框的最小外接矩形
                # TODO 外接矩形处理法
                bboxes[j][0] = min(xtl_1, xtl_2)
                bboxes[j][1] = min(ytl_1, ytl_2)
                bboxes[j][2] = max(xbr_1, xbr_2)
                bboxes[j][3] = max(ybr_1, ybr_2)
                bboxes[j][-1] = max(bboxes[i][-1], bboxes[j][-1])
                boxes[j] = bboxes[j][:4]
                flag = 0
                break
        if flag:  # 该检测框被保留
            keep.append(i)
    bboxes = bboxes[keep]
    labels = labels[keep]
    return bboxes, labels


def post_newnms(bboxes, labels):
    """
        参数 ： predictions:经过置信度筛选后的BoxList
        返回 : 合并具有高度重叠的框后BoxList
        """
    scores = bboxes[:, -1]
    boxes = bboxes[:, :4]

    keep = []
    abandon = []  # 被抛弃的框的索引
    length = len(bboxes)
    if length == 0:
        return bboxes, labels
    for i in range(length):
        if i in abandon:
            continue
        box1 = boxes[i]
        top_left1, bottom_right1 = box1[:2].tolist(), box1[2:].tolist()  # 左上角与右下角坐标
        xtl_1, ytl_1 = top_left1
        xbr_1, ybr_1 = bottom_right1
        area1 = (xbr_1 - xtl_1) * (ybr_1 - ytl_1)
        flag = 1
        for j in range(i + 1, length):
            box2 = boxes[j]
            top_left2, bottom_right2 = box2[:2].tolist(), box2[2:].tolist()  # 左上角与右下角坐标
            xtl_2, ytl_2 = top_left2
            xbr_2, ybr_2 = bottom_right2
            area2 = (xbr_2 - xtl_2) * (ybr_2 - ytl_2)

            intersection, IOU = box_iou(xtl_1, ytl_1, xbr_1, ybr_1, xtl_2, ytl_2, xbr_2, ybr_2)
            if IOU == 0:
                continue
            if IOU > 0.5:  # 两个框的重叠度过高 抑制分类分数小的框
                if scores[i] < scores[j]:
                    flag = 0
                    break
                elif scores[i] == scores[j]:  # 分数相同时 抑制面积小的
                    if area1 < area2:
                        flag = 0
                        break
                    else:
                        abandon.append(j)
                else:
                    abandon.append(j)
        if flag:  # 该检测框被保留
            keep.append(i)
    bboxes = bboxes[keep]
    labels = labels[keep]
    return bboxes, labels


def post_small(bboxes, labels):
    """
        参数 ： predictions:经过置信度筛选后的BoxList
        返回 : 合并具有高度重叠的框后BoxList
        """
    scores = bboxes[:, -1]
    boxes = bboxes[:, :4]

    keep = []
    abandon = []  # 被抛弃的框的索引
    length = len(bboxes)
    if length == 0:
        return bboxes, labels
    for i in range(length):
        if i in abandon:
            continue
        box1 = boxes[i]
        top_left1, bottom_right1 = box1[:2].tolist(), box1[2:].tolist()  # 左上角与右下角坐标
        xtl_1, ytl_1 = top_left1
        xbr_1, ybr_1 = bottom_right1
        area1 = (xbr_1 - xtl_1) * (ybr_1 - ytl_1)
        flag = 1
        for j in range(i + 1, length):
            box2 = boxes[j]
            top_left2, bottom_right2 = box2[:2].tolist(), box2[2:].tolist()  # 左上角与右下角坐标
            xtl_2, ytl_2 = top_left2
            xbr_2, ybr_2 = bottom_right2
            area2 = (xbr_2 - xtl_2) * (ybr_2 - ytl_2)

            intersection, IOU = box_iou(xtl_1, ytl_1, xbr_1, ybr_1, xtl_2, ytl_2, xbr_2, ybr_2)
            # TODO 是否需要考虑类别
            # if labels[i] != labels[j]:
            #     continue
            if IOU == 0:
                continue
            if IOU:  # 当两个框重叠时 小框的大部分区域与大框重叠 抑制小框 TODO 可以有的更好的策略
                if (intersection / area1) > 0.75 and area1 < area2:
                    flag = 0
                    break
                elif (intersection / area2) > 0.75 and area2 < area1:
                    abandon.append(j)
        if flag:  # 该检测框被保留
            keep.append(i)
    bboxes = bboxes[keep]
    labels = labels[keep]
    return bboxes, labels
