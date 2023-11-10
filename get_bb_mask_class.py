# python your_script.py --config_file /path/to/config.yaml --weights /path/to/weights.pth --input_video /path/to/video.mp4 --output_dir /path/to/output/dir


import argparse
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Set up the command line arguments
parser = argparse.ArgumentParser(
    description="Perform instance segmentation on a video using Detectron2 with Mask R-CNN and save the results.")
parser.add_argument("--config_file", help="Path to the config file.", required=True)
parser.add_argument("--weights", help="Path to the weights file.", required=True)
parser.add_argument("--input_video", help="Path to the input video file.", required=True)
parser.add_argument("--output_dir", help="Path to the directory to save the results.", required=True)
args = parser.parse_args()

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = args.weights

# Set up the predictor
predictor = DefaultPredictor(cfg)

# Open your video file
cap = cv2.VideoCapture(args.input_video)

frame_id = 0
while True:
    # Read the next frame
    ret, frame = cap.read()

    # If the frame is not valid, break the loop
    if not ret:
        break

    # Make prediction
    outputs = predictor(frame)

    # Extract instance masks, bounding boxes, and class information
    instances = outputs["instances"]
    masks = instances.pred_masks.to("cpu").numpy()
    boxes = instances.pred_boxes.tensor.to("cpu").numpy()
    classes = instances.pred_classes.to("cpu").numpy()

    # Save masks, boxes, and classes
    np.save(f"{args.output_dir}/masks_{frame_id}.npy", masks)
    np.save(f"{args.output_dir}/boxes_{frame_id}.npy", boxes)
    np.save(f"{args.output_dir}/classes_{frame_id}.npy", classes)

    frame_id += 1

cap.release()
