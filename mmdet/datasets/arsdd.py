from .f1coco import F1ScoreCOCO
from mmdet.datasets.builder import DATASETS

@DATASETS.register_module()
class ARSDDDataset(F1ScoreCOCO):
    CLASSES = ('ZXLF', 'HXLF', 'GFXB', 'KZXB', 'KC', 'KZLF')