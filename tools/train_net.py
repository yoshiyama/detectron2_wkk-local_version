#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import cv2
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import register_custom_dataset #This is a custom script I created.

from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader

# # データセットのルートディレクトリを指定してデータセットを登録
# dataset_root = "./datasets"  # ここに適切なパスを指定してください
# register_custom_dataset.register_custom_datasets(dataset_root)

from detectron2.utils.visualizer import ColorMode


class MyCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._dataset_name = dataset_name  # 追加する行

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            original_height, original_width = input["height"], input["width"]

            img = input["image"].numpy().transpose(1, 2, 0)
            img = cv2.resize(img, (original_width, original_height))  # 元のサイズに戻す

            v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(self._dataset_name))
            v = v.draw_instance_predictions(output["instances"].to("cpu"))
            vis_img = v.get_image()[:, :, ::-1]

            img_name = input["file_name"].split("/")[-1]
            vis_path = os.path.join(self._output_dir, img_name)
            cv2.imwrite(vis_path, vis_img)

        # 親クラスの処理を呼び出す
        super().process(inputs, outputs)
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     return build_evaluator(cfg, dataset_name, output_folder)
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # MyCOCOEvaluator を使用するように変更
        return MyCOCOEvaluator(dataset_name, cfg, True, output_folder)

    def test_with_visualization(self):
        evaluator = self.build_evaluator(
            self.cfg, self.cfg.DATASETS.TEST[0], os.path.join(self.cfg.OUTPUT_DIR, "visualization")
        )
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        # inference_on_dataset(self.model, val_loader, evaluator)
        # Now, evaluator will have the visualization results that can be processed.
        results = inference_on_dataset(self.model, val_loader, evaluator)
        # Save the visualization results if necessary
        return results


    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

# from detectron2.engine import default_argument_parser


def main(args):
    cfg = setup(args)
    dataset_root = args.dataset_root  # コマンドライン引数から取得
    register_custom_dataset.register_custom_datasets(dataset_root)  # データセットを登録

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset_root", default="./datasets", help="Root directory of the dataset")

    # args = default_argument_parser().parse_args()
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )