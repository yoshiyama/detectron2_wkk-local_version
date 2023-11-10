from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
import cv2
import os

# Initialize the config
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("root41_train",)
cfg.DATASETS.TEST = ("root41_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = "/home/survey/detectron2_wkk/tools/checkpoints/model_bridge"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

# Directory paths
input_dir = "/home/survey/keikan_bridge/kiritori-henkan1/JPEGImages"
output_dir = "/home/survey/keikan_bridge/kiritori-henkan1/JPEGImages_cut"

# Get a list of image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    # Load an image
    image_path = os.path.join(input_dir, image_file)
    img = cv2.imread(image_path)

    # Make prediction
    outputs = predictor(img)

    # Get the instances from the prediction
    instances = outputs["instances"].to("cpu")

    # Get the bounding box coordinates for each instance
    for i, box in enumerate(instances.pred_boxes):
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = box.tolist()

        # Crop the image based on the bounding box coordinates
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

        # Save the cropped image with a modified file name
        cropped_file = image_file.replace(".", "_cropped.")
        output_path = os.path.join(output_dir, cropped_file)
        cv2.imwrite(output_path, cropped_img)

