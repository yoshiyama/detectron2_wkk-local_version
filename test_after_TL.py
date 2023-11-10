from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
import cv2
import os

# Initialize the config
cfg = get_cfg()

# Here, add your own settings that you used during training, for example:
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("root41_train",)
cfg.DATASETS.TEST = ("root41_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = "/home/survey/detectron2_wkk/tools/checkpoints/model_bridge"

# Set up the predictor
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

# Load an image
img = cv2.imread("/home/survey/keikan_bridge/kiritori-henkan3/JPEGImages/katoA040.jpg")

# Make prediction
outputs = predictor(img)

# Visualize the prediction
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('predictions', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
