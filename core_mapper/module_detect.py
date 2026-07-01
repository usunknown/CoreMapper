"""
Module 2: 特征识别 — YOLO 多模型推理 + 坐标映射
"""
import json
import os
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def load_models(model_configs):
    """
    加载多个 YOLO 模型。
    model_configs: [{"path": "...", "classes": ["crack"]}, ...]
    返回 [(model, class_list), ...]
    """
    if not HAS_YOLO:
        raise ImportError("未安装 ultralytics: pip install ultralytics")

    models = []
    for cfg in model_configs:
        model = YOLO(cfg["path"])
        models.append((model, cfg["classes"]))
    return models


def detect_on_rectified(rectified_img, calib_data, models, conf_threshold=0.25):
    """
    在校正后的图像上运行多模型 YOLO 推理。
    calib_data: 标定字典 {"corners": [...], "rows": 6, "depth_start": 24.0, "depth_end": 30.0}
    返回 detections 列表。
    """
    h, w = rectified_img.shape[:2]
    all_dets = []

    for model, class_list in models:
        results = model(rectified_img, conf=conf_threshold, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf_val, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                depth = _rect_to_depth(cx, cy, w, h, calib_data)
                cls_name = class_list[cls_id] if cls_id < len(class_list) else str(cls_id)

                all_dets.append({
                    "class": cls_name,
                    "confidence": round(float(conf_val), 4),
                    "depth": round(float(depth), 3),
                    "bbox": [round(float(v), 1) for v in bbox],
                    "center": [round(float(cx), 1), round(float(cy), 1)],
                })

    return all_dets


def detect_on_directory(image_dir, model_configs, conf_threshold=0.25,
                        progress_callback=None):
    """
    批量推理目录下所有 _rectified.jpg，或对原图先标定再推理。
    progress_callback(current, total) 返回 True 表示取消。
    返回 {image_name: [detections, ...], ...}
    """
    if not HAS_YOLO:
        raise ImportError("未安装 ultralytics: pip install ultralytics")

    models = load_models(model_configs)
    jpgs = sorted([f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg'))
                   and "_rectified" not in f
                   and "_annotated" not in f])

    results = {}
    for i, jpg in enumerate(jpgs):
        path = os.path.join(image_dir, jpg)

        # 尝试加载标定
        json_path = os.path.splitext(path)[0] + "_calib.json"
        if not os.path.exists(json_path):
            continue  # 跳过未标定的图片

        with open(json_path, "r", encoding="utf-8") as f:
            calib = json.load(f)

        # 尝试加载已校正图，否则现场校正
        rect_path = os.path.splitext(path)[0] + "_rectified.jpg"
        if os.path.exists(rect_path):
            img = cv2.imread(rect_path)
        else:
            from .module_rectify import rectify
            img, _, _ = rectify(path, calib)

        dets = detect_on_rectified(img, calib, models, conf_threshold)
        results[jpg] = dets

        # 保存检测 JSON
        det_path = os.path.splitext(path)[0] + "_detections.json"
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(dets, f, ensure_ascii=False, indent=2)

        if progress_callback:
            if progress_callback(i + 1, len(jpgs)):
                break

    return results


def _rect_to_depth(cx, cy, img_w, img_h, calib):
    """
    校正后矩形坐标 → 深度值。
    假设 row_layout="snake"（蛇形排列）：
    第 0 排左→右深度递增，第 1 排右→左深度递增...
    """
    rows = calib.get("rows", 6)
    d_start = calib.get("depth_start", 0)
    d_end = calib.get("depth_end", rows * 1.0)
    layout = calib.get("row_layout", "snake")

    row_h = img_h / rows
    row_idx = int(cy // row_h)
    row_idx = max(0, min(row_idx, rows - 1))
    col_ratio = cx / img_w
    col_ratio = max(0.0, min(col_ratio, 1.0))

    if layout == "snake" and row_idx % 2 == 1:
        col_ratio = 1.0 - col_ratio  # 反向

    depth_per_row = (d_end - d_start) / rows
    depth = d_start + (row_idx + col_ratio) * depth_per_row
    return depth
