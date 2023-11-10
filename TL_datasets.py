from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

register_coco_instances("root41", {"thing_classes": ["bridge"]}, "/mnt/c/Users/survey/Desktop/kiritori-henkan/annotations.json", "/mnt/c/Users/survey/Desktop/kiritori-henkan/JPEGImages")

root41_metadata = MetadataCatalog.get("root41")
dataset_dicts = DatasetCatalog.get("root41")

print(root41_metadata)

# データセットの情報を表示する例
for d in dataset_dicts:
    print(d["file_name"])  # 画像パス
    print(d["annotations"])  # アノテーション情報
