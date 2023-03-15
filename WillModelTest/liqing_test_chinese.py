import glob
import json
import logging
import os

from tqdm import tqdm

from WillEvaluate.liqing_evaluate_diagonal import liqing_evaluate
from WillModelTest.liqing_show_chinese import show_result
from mmdet.apis import init_detector, inference_detector

CLASSES = ('纵向裂缝', '横向裂缝', '灌缝修补', '块状修补', '坑槽', '块状裂缝')


def test():
    for img_name in tqdm(file_list):
        img_path = file_root + img_name
        result = inference_detector(model, img_path)
        save_path = save_out + img_name
        # save_path = None

        bboxes, labels = show_result(img_path, result, CLASSES, score_thr=0.0, out_file=save_path)

        # 存储预测结果(回归坐标+类别)
        prediction_json = {"bbox": bboxes[:, :4].tolist(),
                           "labels": labels.tolist()
                           }
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(json_output_folder, json_name)

        # 0.00006s
        with open(json_path, 'w') as json_file:
            json.dump(prediction_json, json_file, indent=4)


if __name__ == "__main__":

    # TODO 根據配置文件修改
    config_path = '/home/will/mmdetection/work_dirs/lq_faster_rcnn_res50_sefpn_roiExtract'
    config_file = config_path + '/' + os.path.basename(config_path) + '.py'
    pth_file = config_path + "/*.pth"
    all_pth = sorted(glob.glob(pth_file), key=os.path.getmtime)

    # 测试文件图片文件夹
    file_root = '/home/will/mmdetection/data/coco/val2017/'
    file_list = os.listdir(file_root)
    FileLength = len(file_list)
    # 测试图片结果保存文件夹
    save_out = "/home/WillEnglishPaper/mmdetection/liqing_result/"
    # 测试结果保存文件夹
    json_output_folder = "/home/WillEnglishPaper/mmdetection/prediction_json/liqing"

    # 打印日志信息
    logger = logging.getLogger("eval_log")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(config_path + "/liqing_eval_log_v5.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    for pth in all_pth:
        logger.info(pth)
        checkpoint_file = pth
        # build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        test()
        logger.info(liqing_evaluate())
