import glob
import json
import logging
import os

from tqdm import tqdm

from demo.rdd_show import show_result
from mmdet.apis import init_detector, inference_detector

CLASSES = ('D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44')


def test():
    for img_name in tqdm(file_list):
        img_path = file_root + img_name
        result = inference_detector(model, img_path)
        save_path = save_out + img_name
        save_path = None

        bboxes, labels = show_result(img_path, result, CLASSES, score_thr=0.0, out_file=save_path)

        # 存储预测结果(回归坐标+类别)
        prediction_json = {"bbox": bboxes[:, :4].tolist(),
                           "labels": labels.tolist()
                           }
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(json_output_folder, json_name)
        #
        # # 0.00006s
        with open(json_path, 'w') as json_file:
            json.dump(prediction_json, json_file, indent=4)


if __name__ == "__main__":

    # TODO 根據配置文件修改
    config_path = '/home/will/mmdetection/work_dirs/rdd2018_faster_rcnn_cres50_sefpn_iouatss_ncl_roiExtrac'
    config_file = config_path + '/' + os.path.basename(config_path) + '.py'
    pth_file = config_path + "/*.pth"
    all_pth = sorted(glob.glob(pth_file), key=os.path.getmtime)

    # 测试文件图片文件夹
    file_root = '/home/will/mmdetection/RDD2018Test/'
    file_list = os.listdir(file_root)
    FileLength = len(file_list)
    # 测试图片结果保存文件夹
    save_out = "/home/WillEnglishPaper/mmdetection/work_dirs/rdd2018_faster_rcnn_cres50_sefpn_iouatss_ncl_roiExtrac/images/"
    # 测试结果保存文件夹
    json_output_folder = "/home/WillEnglishPaper/mmdetection/RDD2018TestJson"

    for pth in all_pth:
        checkpoint_file = pth
        # build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        test()
