# register_cusstom_dataset.py
# Below is the content of a sample Python file for registering a dataset in the COCO format with Detectron2:
# By executing this register_dataset.py file, a custom dataset gets registered with Detectron2. By changing the dataset_root_path to the appropriate directory, you can specify the location of the dataset according to your environment.

# Usage:python register_custom_dataset.py --dataset_root ./datasets

# After running this script, you can execute train_net.py to start the training.


from detectron2.data.datasets import register_coco_instances
import os
import argparse


def register_custom_datasets(dataset_root):
    """
    Register custom datasets in COCO format.

    Parameters:
    - dataset_root (str): Root directory where the datasets are stored.
    """

    # Train dataset
    train_image_dir = os.path.join(dataset_root, "train/JPEGImages")
    train_annotation_file = os.path.join(dataset_root, "train/annotations.json")
    register_coco_instances("custom_dataset_train", {}, train_annotation_file, train_image_dir)

    # Validation dataset
    val_image_dir = os.path.join(dataset_root, "val/JPEGImages")
    val_annotation_file = os.path.join(dataset_root, "val/annotations.json")
    register_coco_instances("custom_dataset_val", {}, val_annotation_file, val_image_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register custom datasets in COCO format with Detectron2.")
    parser.add_argument("--dataset_root", required=True, help="Root directory where the datasets are stored.")

    args = parser.parse_args()

    # Register the datasets
    register_custom_datasets(args.dataset_root)
    print("Datasets registered!")