import os

from tqdm import tqdm

from demo.liqing_show import show_result
from mmdet.apis import init_detector, inference_detector

CLASSES = ('D00', 'D10', 'D20', 'D40')


def test():
    result_list = []
    for img_name in tqdm(file_list):
        img_path = file_root + img_name
        result = inference_detector(model, img_path)
        save_path = save_out + img_name
        res = img_name + ","
        save_path = None
        bboxes, labels = show_result(img_path, result, CLASSES, score_thr=0.05, out_file=save_path)

        bboxes = bboxes[:, :4].tolist()
        labels = labels.tolist()
        for bbox, label in zip(bboxes, labels):
            res = res + str(label + 1) + " " + str(int(bbox[0])) + " " + str(int(bbox[1])) + " " + str(
                int(bbox[2])) + " " + str(int(bbox[3])) + " "
        result_list.append(res)
    # TODO 写入的提交文件(.txt)
    f = open("/home/will/mmdetection/submission/test2-2.txt", mode="a")

    for res in result_list:
        f.write(res + "\n")
    f.close()


if __name__ == "__main__":
    # TODO 配置文件和权重文件
    config_file = "/configs/WillEnglishPaper/rdd2020_faster_rcnn_sefpn_rpnatss_1anchor_res50.py"
    pth_file = "/home/will/mmdetection/work_dirs/rdd2020_faster_rcnn_sefpn_rpnatss_1anchor_res50/latest.pth"

    # TODO 测试文件图片文件夹
    file_root = '/home/will/mmdetection/RDD2020Test/test2/Czech/images/'
    file_list = os.listdir(file_root)
    FileLength = len(file_list)
    # 测试图片结果保存文件夹
    save_out = "/home/WillEnglishPaper/mmdetection/submission/"

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, pth_file, device='cuda:0')
    test()
