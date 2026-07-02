"""
Module 3: 特征位置数据库 — 文件名解析 → 深度映射 → CSV 导出
"""
import csv
import json
import os
import re


def parse_filename(filename):
    """
    从文件名提取钻孔信息。
    "CK24-24.0-30.0m.jpg" → {"hole_id": "CK24", "depth_start": 24.0, "depth_end": 30.0}
    "CK24-24.0-30.0m_rectified.jpg" → 同上（去掉后缀）
    """
    name = os.path.splitext(filename)[0]
    # 去掉 _rectified, _annotated 等后缀
    name = re.sub(r'_(rectified|annotated|calib|detections|depth)$', '', name)
    # 匹配模式: CK{数字}-{数字}-{数字}m
    m = re.match(r'(CK\d+)-(\d+\.?\d*)-(\d+\.?\d*)m', name)
    if m:
        hole_id = m.group(1)
        depth_start = float(m.group(2))
        depth_end = float(m.group(3))
        rows = int(depth_end - depth_start)
        return {"hole_id": hole_id, "depth_start": depth_start,
                "depth_end": depth_end, "rows": max(rows, 1)}
    return None


def _fallback_info(directory, base):
    """从 calib JSON 和文件名猜深度信息"""
    calib_path = os.path.join(directory, base + "_calib.json")
    if not os.path.exists(calib_path):
        return None
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    # 从文件名提取孔号（如 IMG_1816 → "IMG"）
    import re
    m = re.match(r'(CK\d+)', base)
    hole_id = m.group(1) if m else base
    return {
        "hole_id": hole_id,
        "depth_start": calib.get("depth_start", 0),
        "depth_end": calib.get("depth_end", 1),
        "rows": calib.get("rows", 1),
    }


def collect_detections(directory, progress_callback=None):
    """
    扫描 detections/*/ 子目录，汇总为记录列表。
    """
    import glob

    det_base = os.path.join(directory, "detections")
    if not os.path.exists(det_base):
        return []

    records = []
    # 收集所有 _detections.json
    det_files = sorted(glob.glob(os.path.join(det_base, "*", "*_detections.json")))
    if not det_files:
        # 向下兼容：旧格式（平铺在目录下）
        det_files = sorted(glob.glob(os.path.join(directory, "*_detections.json")))

    for i, det_path in enumerate(det_files):
        # 从路径提取 base 名和 class
        fname = os.path.basename(det_path)
        cls_name = os.path.basename(os.path.dirname(det_path))
        if cls_name == "detections":
            cls_name = "unknown"
        base = fname.replace("_detections.json", "")

        info = parse_filename(base)
        if info is None:
            info = _fallback_info(directory, base)
        if info is None:
            continue

        # 查找原图
        jpg = base + ".jpg" if os.path.exists(os.path.join(directory, base + ".jpg")) else None
        jpg = jpg or (base + ".JPG" if os.path.exists(os.path.join(directory, base + ".JPG")) else None)
        if jpg is None:
            continue

        with open(det_path, "r", encoding="utf-8") as f:
            detections = json.load(f)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = d.get("center", [(x1 + x2) / 2, (y1 + y2) / 2])
            records.append({
                "hole_id": info["hole_id"],
                "depth_m": d.get("depth", 0),
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_x2": x2, "bbox_y2": y2,
                "center_x": cx, "center_y": cy,
                "image_file": jpg,
            })

        if progress_callback:
            progress_callback(i + 1, len(det_files))

    return records


def export_csv(records, output_path):
    """导出 CSV"""
    fieldnames = ["hole_id", "depth_m", "class", "confidence",
                  "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                  "center_x", "center_y", "image_file"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["hole_id"], r["depth_m"])):
            writer.writerow(rec)

    return output_path


def export_json(records, output_path):
    """导出 JSON（按钻孔分组）"""
    grouped = {}
    for rec in records:
        hid = rec["hole_id"]
        if hid not in grouped:
            grouped[hid] = []
        grouped[hid].append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    return output_path


# ------------------------------------------------------------------
# 预留接口：深度校准
# ------------------------------------------------------------------

def calibrate_depth_offset(core_features_csv, televiewer_features_csv,
                           anchor_points=None, output_csv=None):
    """
    预留接口：岩芯-电视深度校准。

    输入:
        core_features_csv: 岩芯特征 CSV 路径
        televiewer_features_csv: 电视特征 CSV 路径
        anchor_points: [(core_depth, tv_depth), ...] 手动锚点对

    输出:
        偏移后的特征 CSV

    阶段二实现。
    """
    raise NotImplementedError(
        "深度校准模块将在阶段二实现。请手动提供 anchor_points。"
    )
