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
    返回按 class 分组的 detections 字典: {"crack": [...], "intrusion": [...]}
    """
    h, w = rectified_img.shape[:2]
    grouped = {}

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

                det = {
                    "class": cls_name,
                    "confidence": round(float(conf_val), 4),
                    "depth": round(float(depth), 3),
                    "bbox": [round(float(v), 1) for v in bbox],
                    "center": [round(float(cx), 1), round(float(cy), 1)],
                }
                grouped.setdefault(cls_name, []).append(det)

    return grouped


def detect_on_directory(image_dir, model_configs, conf_threshold=0.25,
                        progress_callback=None):
    """
    批量推理目录下所有原图。
    检测结果写入 detections/{class_name}/{name}_detections.json
    """
    if not HAS_YOLO:
        raise ImportError("未安装 ultralytics: pip install ultralytics")

    models = load_models(model_configs)
    rect_dir = os.path.join(image_dir, "rectified")

    jpgs = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
        and "_review" not in f
        and "_annotated" not in f
    ])

    total_dets = 0
    for i, jpg in enumerate(jpgs):
        name = os.path.splitext(jpg)[0]
        path = os.path.join(image_dir, jpg)

        # 加载标定
        calib_path = os.path.join(image_dir, name + "_calib.json")
        if not os.path.exists(calib_path):
            continue

        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)

        # 加载校正图
        rect_path = os.path.join(rect_dir, name + "_rectified.jpg")
        if not os.path.exists(rect_path):
            from .module_rectify import rectify
            img, _, _ = rectify(path, calib)
        else:
            img = cv2.imread(rect_path)

        grouped = detect_on_rectified(img, calib, models, conf_threshold)

        # 按类别分文件保存
        for cls_name, dets in grouped.items():
            det_dir = os.path.join(image_dir, "detections", cls_name)
            os.makedirs(det_dir, exist_ok=True)
            det_path = os.path.join(det_dir, name + "_detections.json")
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(dets, f, ensure_ascii=False, indent=2)
            total_dets += len(dets)

        if progress_callback:
            if progress_callback(i + 1, len(jpgs)):
                break

    return total_dets


def _rect_to_depth(cx, cy, img_w, img_h, calib):
    """校正后矩形坐标 → 深度值（蛇形排列）"""
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
        col_ratio = 1.0 - col_ratio

    depth_per_row = (d_end - d_start) / rows
    depth = d_start + (row_idx + col_ratio) * depth_per_row
    return depth


# ================================================================
# TV 图像推理管线
# ================================================================

def detect_tv_directory(tv_dir, model_configs, conf_threshold=0.25,
                         progress_callback=None):
    """
    TV 图像批量推理：
    - 从 tv_calib.json 读取有效区域并裁剪
    - 从文件名解析深度信息
    - 检测结果写入 tv_dir/detections/{class_name}/{name}_detections.json
    """
    if not HAS_YOLO:
        raise ImportError("unable to import ultralytics: pip install ultralytics")

    from .module_tv_calib import load_tv_calib, crop_tv_image
    from .module_tv_parse import parse_tv_filename

    calib = load_tv_calib(tv_dir)
    if calib is None:
        print(f"TV calibration not found: {tv_dir}/tv_calib.json")
        return 0

    models = load_models(model_configs)

    jpgs = sorted([
        f for f in os.listdir(tv_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and "_review" not in f
        and "_mask" not in f
        and "_annotated" not in f
    ])

    total_dets = 0
    for i, fname in enumerate(jpgs):
        path = os.path.join(tv_dir, fname)
        info = parse_tv_filename(fname)
        if info is None:
            continue

        # 裁剪 + 推理
        img = crop_tv_image(path, calib)
        if img is None:
            continue

        h, w = img.shape[:2]
        calib_for_detect = {
            "rows": 1,
            "depth_start": info["z_top"],
            "depth_end": info["z_bottom"],
            "row_layout": "snake",
        }

        grouped = detect_on_rectified(img, calib_for_detect, models,
                                      conf_threshold)

        # 按类别分文件保存
        name = os.path.splitext(fname)[0]
        for cls_name, dets in grouped.items():
            det_dir = os.path.join(tv_dir, "detections", cls_name)
            os.makedirs(det_dir, exist_ok=True)
            det_path = os.path.join(det_dir, name + "_detections.json")
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(dets, f, ensure_ascii=False, indent=2)
            total_dets += len(dets)

        if progress_callback:
            if progress_callback(i + 1, len(jpgs)):
                break

    return total_dets
