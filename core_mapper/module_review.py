"""
Module Review: Labelme JSON ↔ detections 双向转换
"""
import json
import os


def detections_to_labelme(rectified_path, detections, calib_data, output_path=None):
    """
    检测结果 → Labelme JSON（用于 labelme 人工审核）。
    rectified_path: 校正图的完整路径
    detections: [{"class":..., "confidence":..., "bbox":..., "depth":...}, ...]
    calib_data: 标定 JSON 内容

    矩形框转为 polygon 四个顶点，置信度和深度保存在 shape 的 flags 中。
    """
    import cv2

    # 读取校正图尺寸
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
            "flags": {
                "confidence": det.get("confidence", 0),
                "depth": det.get("depth", 0),
                "det_id": i,  # 跟踪原始检测 ID
            }
        }
        shapes.append(shape)

    # 审核 JSON 对应的图是 _review.jpg
    review_jpg = os.path.splitext(os.path.basename(rectified_path))[0].replace("_rectified", "") + "_review.jpg"

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
        output_path = os.path.splitext(rectified_path)[0] + "_review.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    return output_path


def labelme_to_detections(labelme_path, calib_data=None):
    """
    Labelme JSON → 检测结果列表。
    读取用户审核后的 labelme JSON，提取所有 shape 的外接矩形作为 bbox。
    """

    with open(labelme_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = []
    for i, shape in enumerate(data.get("shapes", [])):
        if not shape.get("points"):
            continue
        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        flags = shape.get("flags", {})

        det = {
            "class": shape["label"],
            "confidence": flags.get("confidence", 1.0),
            "depth": flags.get("depth", 0),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            "reviewed": True,  # 标记为已审核
        }
        detections.append(det)

    return detections


def export_all_for_review(image_dir, progress_callback=None):
    """目录下所有 _detections.json 批量导出为 _review.json"""
    import glob
    det_files = sorted(glob.glob(os.path.join(image_dir, "*_detections.json")))
    exported = 0
    for i, det_path in enumerate(det_files):
        base = os.path.basename(det_path).replace("_detections.json", "")
        rect_path = os.path.join(image_dir, base + "_rectified.jpg")
        calib_path = os.path.join(image_dir, base + "_calib.json")
        if not os.path.exists(rect_path) or not os.path.exists(calib_path):
            continue
        with open(det_path, "r", encoding="utf-8") as f:
            detections = json.load(f)
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        output = os.path.join(image_dir, base + "_review.json")
        detections_to_labelme(rect_path, detections, calib, output)
        # 复制校正图，带 _review 后缀，labelme 保存时会自动生成 _review.json
        import shutil
        review_img = os.path.join(image_dir, base + "_review.jpg")
        if not os.path.exists(review_img):
            shutil.copy2(rect_path, review_img)
        exported += 1
        if progress_callback:
            progress_callback(i + 1, len(det_files))
    return exported


def import_all_reviewed(image_dir, progress_callback=None):
    """目录下所有 _review.json 批量回读，覆盖 _detections.json"""
    import glob
    review_files = sorted(glob.glob(os.path.join(image_dir, "*_review.json")))
    imported = 0
    for i, rv_path in enumerate(review_files):
        dets = labelme_to_detections(rv_path)
        base = os.path.basename(rv_path).replace("_review.json", "")
        det_path = os.path.join(image_dir, base + "_detections.json")
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(dets, f, ensure_ascii=False, indent=2)
        imported += 1
        if progress_callback:
            progress_callback(i + 1, len(review_files))
    return imported
