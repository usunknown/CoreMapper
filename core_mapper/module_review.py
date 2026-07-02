"""
Module Review: Labelme JSON ↔ detections 双向转换
"""
import json
import os
import shutil


def detections_to_labelme(rectified_path, detections, output_path=None):
    """
    检测结果 → Labelme JSON。
    rectified_path: 校正图的完整路径
    detections: [{"class":..., "confidence":..., "bbox":..., "depth":...}, ...]

    置信度和深度保存在 shape 的 description 字段中（JSON 字符串）。
    """
    import cv2

    img = cv2.imread(rectified_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {rectified_path}")
    h, w = img.shape[:2]

    shapes = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
        shape = {
            "label": det["class"],
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": json.dumps({
                "confidence": det.get("confidence", 0),
                "depth": det.get("depth", 0),
                "det_id": i,
            }),
        }
        shapes.append(shape)

    # imagePath 指向同目录下的 review jpg
    review_jpg = os.path.basename(rectified_path).replace("_rectified", "_review")

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": review_jpg,
        "imageHeight": h,
        "imageWidth": w,
        "imageData": None,
    }

    if output_path is None:
        output_path = rectified_path.replace("_rectified.jpg", "_review.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    return output_path


def labelme_to_detections(labelme_path):
    """
    Labelme JSON → 检测结果列表。
    读取用户审核后的 labelme JSON，提取所有 shape 的外接矩形作为 bbox。
    """
    with open(labelme_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = []
    for shape in data.get("shapes", []):
        if not shape.get("points"):
            continue
        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        meta = {}
        desc = shape.get("description", "")
        if desc:
            try:
                meta = json.loads(desc)
            except (json.JSONDecodeError, ValueError):
                pass

        det = {
            "class": shape["label"],
            "confidence": meta.get("confidence", 1.0),
            "depth": meta.get("depth", 0),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            "reviewed": True,
        }
        detections.append(det)

    return detections


def export_all_for_review(image_dir, class_name=None, progress_callback=None):
    """
    目录下所有 detections/{class}/_detections.json → review/{class}/_review.json + _review.jpg
    如果 class_name 为 None，处理所有类别。
    """
    import glob

    det_base = os.path.join(image_dir, "detections")
    if not os.path.exists(det_base):
        return 0

    if class_name:
        classes = [class_name]
    else:
        classes = sorted(
            d for d in os.listdir(det_base)
            if os.path.isdir(os.path.join(det_base, d))
        )

    exported = 0
    for cls in classes:
        det_dir = os.path.join(det_base, cls)
        rect_dir = os.path.join(image_dir, "rectified")
        calib_dir = image_dir
        review_dir = os.path.join(image_dir, "review", cls)
        os.makedirs(review_dir, exist_ok=True)

        det_files = sorted(glob.glob(os.path.join(det_dir, "*_detections.json")))
        for det_path in det_files:
            base = os.path.basename(det_path).replace("_detections.json", "")
            rect_path = os.path.join(rect_dir, base + "_rectified.jpg")
            calib_path = os.path.join(calib_dir, base + "_calib.json")
            if not os.path.exists(rect_path) or not os.path.exists(calib_path):
                continue
            with open(det_path, "r", encoding="utf-8") as f:
                detections = json.load(f)
            with open(calib_path, "r", encoding="utf-8") as f:
                pass  # calib not needed for export

            output = os.path.join(review_dir, base + "_review.json")
            detections_to_labelme(rect_path, detections, output)

            # 复制校正图带 _review 后缀 → labelme 自动保存 _review.json
            review_jpg = os.path.join(review_dir, base + "_review.jpg")
            if not os.path.exists(review_jpg):
                shutil.copy2(rect_path, review_jpg)
            exported += 1

    return exported


def import_all_reviewed(image_dir, class_name=None, progress_callback=None):
    """
    review/{class}/_review.json → 覆盖 detections/{class}/_detections.json
    """
    import glob

    review_base = os.path.join(image_dir, "review")
    if not os.path.exists(review_base):
        return 0

    if class_name:
        classes = [class_name]
    else:
        classes = sorted(
            d for d in os.listdir(review_base)
            if os.path.isdir(os.path.join(review_base, d))
        )

    imported = 0
    for cls in classes:
        review_dir = os.path.join(review_base, cls)
        det_dir = os.path.join(image_dir, "detections", cls)
        os.makedirs(det_dir, exist_ok=True)

        review_files = sorted(glob.glob(os.path.join(review_dir, "*_review.json")))
        for rv_path in review_files:
            dets = labelme_to_detections(rv_path)
            base = os.path.basename(rv_path).replace("_review.json", "")
            det_path = os.path.join(det_dir, base + "_detections.json")
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(dets, f, ensure_ascii=False, indent=2)
            imported += 1

    return imported
