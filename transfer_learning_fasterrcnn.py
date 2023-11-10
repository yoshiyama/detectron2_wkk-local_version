from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# register_coco_instances("root41", {"thing_classes": ["bridge"]}, "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan1/annotations.json", "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan1/JPEGImages")
register_coco_instances("root41_train", {}, "/home/survey/keikan_bridge/kiritori-henkan1/annotations.json", "/home/survey/keikan_bridge/kiritori-henkan1/")
register_coco_instances("root41_val", {}, "/home/survey/keikan_bridge/kiritori-henkan2/annotations.json", "/home/survey/keikan_bridge/kiritori-henkan2/")

root41_metadata = MetadataCatalog.get("root41_train")
dataset_dicts = DatasetCatalog.get("root41_train")

# print(root41_metadata)

# データセットの情報を表示する例
for d in dataset_dicts:
    print(d["file_name"])  # 画像パス
    print(d["annotations"])  # アノテーション情報

root41_metadata = MetadataCatalog.get("root41_val")
dataset_dicts = DatasetCatalog.get("root41_val")

# print(root41_metadata)

# データセットの情報を表示する例
for d in dataset_dicts:
    print(d["file_name"])  # 画像パス
    print(d["annotations"])  # アノテーション情報


from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Faster R-CNNのconfigファイルを指定

cfg.DATASETS.TRAIN = ("root41_train",)
cfg.DATASETS.TEST = ("root41_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "model_weights/model_final_68b088.pkl"  # 転移学習用のモデルを指定
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000    # 適切なiteration数を設定してください。
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # カスタムデータセットのクラス数を指定
cfg.OUTPUT_DIR = "/home/survey/detectron2_wkk/tools/checkpoints/model_bridge"  # 学習結果を保存するディレクトリを指定


from detectron2.engine import DefaultTrainer

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
