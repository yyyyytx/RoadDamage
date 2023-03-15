import os

from demo.shuini_show import show_result
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = '/home/will/mmdetection/work_dirs/liqing_atss_20201029_2/liqing_atss_20201029_2.py'
checkpoint_file = '/home/will/mmdetection/work_dirs/liqing_atss_20201029_2/epoch_21.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

file_root = '/home/will/mmdetection/1030/'
file_list = os.listdir(file_root)
save_out = "/home/WillEnglishPaper/mmdetection/1030res/"
FileLength = len(file_list)
count = 0

for img_name in file_list:
    print(img_name)
    count = count + 1
    print('{}/{}'.format(count, FileLength))
    print('-' * 10)
    img_path = file_root + img_name
    result = inference_detector(model, img_path)
    save_path = save_out + img_name
    show_result(img_path, result, model.CLASSES, score_thr=0.3, out_file=save_path)

# test a single image and show the results
# img = '/home/WillEnglishPaper/mmdetection/shuini_test/G0026497.JPG'  # or img = mmcv.imread(img), which WillEnglishPaper only load it once
# result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# show_result_pyplot(model, img, result, score_thr=0.3)
# or save the visualization results to image files
# model.show_result(img, result, out_file='/home/WillEnglishPaper/mmdetection/shuini_result/result.jpg')
