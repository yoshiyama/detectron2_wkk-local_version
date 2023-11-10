import argparse
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from sklearn import linear_model

# 以下のパラメータを調整してください
POLE_CLASS_ID = 0  # ポールのクラスID
THRESHOLD_SLOPE = 0.1  # 傾きの閾値
COLOR = (0, 0, 255)  # 特定色（BGR）

# コマンドライン引数のパース
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--input_video", required=True)
parser.add_argument("--output_video", required=True)
args = parser.parse_args()

# Detectron2の設定と初期化
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = args.weights
predictor = DefaultPredictor(cfg)

# ビデオの読み込みと書き込みの準備
cap = cv2.VideoCapture(args.input_video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    # フレームの読み込み
    ret, frame = cap.read()
    if not ret:
        break

    # 推論
    outputs = predictor(frame)
    instances = outputs["instances"]

    # ポールのインスタンスのみを抽出
    pole_instances = instances[instances.pred_classes == POLE_CLASS_ID]
    for i in range(len(pole_instances)):
        # マスクから画素の座標を抽出
        mask = pole_instances.pred_masks[i].to("cpu").numpy()
        y, x = np.nonzero(mask)

        # RANSACを用いて線形回帰を実行
        ransac = linear_model.RANSACRegressor()
        ransac.fit(x.reshape(-1, 1), y)
        slope = ransac.estimator_.coef_[0]

        # 傾きが閾値以上であれば、マスクを特定色にする
        if abs(slope) >= THRESHOLD_SLOPE:
            frame[mask] = COLOR

    # フレームを書き込み
    out.write(frame)

cap.release()
out.release
