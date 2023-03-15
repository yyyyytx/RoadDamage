from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class F1ScoreCOCO(CocoDataset):

    def evaluate(self,
                 results,
                 metric='bbox',
                 iou_thrs=None,
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 metric_items=None):
        metrics = metric if isinstance(metric, list) else [metric]
        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        #cal f1 score based on iou 0.5
        iou_thrs=np.array([0.5])
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        print('results:', eval_results)
        return eval_results


    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        eval_results = OrderedDict()
        metric = 'bbox'
        # metric = metrics[0]
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        iou_type = 'bbox'
        try:
            print(result_files, metric)
            predictions = mmcv.load(result_files[metric])
            coco_det = coco_gt.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
            return eval_results

        cocoEval = COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.evaluate()
        cocoEval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        print_log('\n' + redirect_string.getvalue(), logger=logger)
        print(cocoEval.stats)
        mP = cocoEval.stats[1]
        mR = cocoEval.stats[8]
        f1 = 2*mP*mR/(mP+mR+1e-7)
        headers = ['mP', 'mR', 'F1-score']
        table_data = [headers]
        table_data.append([f'{mP:.3f}', f'{mR:.3f}', f'{f1:.3f}'])
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=logger)

        if metric_items is None:
            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = cocoEval.stats[:6]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results

