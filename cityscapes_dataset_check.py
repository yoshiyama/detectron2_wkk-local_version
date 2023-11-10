from detectron2.data import DatasetCatalog

# データセット名
dataset_names = ["cityscapes_fine_instance_seg_train", "cityscapes_fine_instance_seg_val"]

for d in dataset_names:
    # データセットが登録されている場合
    if d in DatasetCatalog.list():
        # データセットの内容を取得
        dataset = DatasetCatalog.get(d)
        # データセットの内容を表示（最初の数件のみ）
        print(dataset[:5])
