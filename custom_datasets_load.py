# What is this program? You might be better off using register_custom_dataset.py instead of this program.
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# register_coco_instances("root41", {"thing_classes": ["bridge"]}, "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan1/annotations.json", "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan1/JPEGImages")
register_coco_instances("root41_train", {}, "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan1/annotations.json", "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan1/JPEGImages")
register_coco_instances("root41_val", {}, "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan2/annotations.json", "/mnt/c/Users/survey/Desktop/keikan_bridge/kiritori-henkan2/JPEGImages")

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
