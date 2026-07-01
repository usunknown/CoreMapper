"""
岩芯照片深度映射工具
- --mode calibrate: 标定岩芯箱四角 + 深度区间 + 行数 → 存 JSON
- --mode detect:    加载标定 JSON + YOLO 检测 → 输出带深度的检测结果
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

# YOLO 为可选依赖，仅在 detect 模式需要
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# ---------------------------------------------------------------------------
# 标定模式
# ---------------------------------------------------------------------------

def calibrate(image_path, depth_start, depth_end, rows):
    """在照片上标定岩芯箱四角，保存标定 JSON"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    corners = []
    window = "Calibration - Click: TL TR BR BL  [ESC=reset ENTER=confirm]"
    # 在图像上显示中文操作提示
    cv2.putText(img, "Click: TL TR BR BL  ESC=reset ENTER=confirm",
                (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    def draw_state():
        disp = img.copy()
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(disp, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(corners) == 4:
            pts = np.array(corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
        # 显示深度信息和行数
        cv2.putText(disp, f"Depth: {depth_start} ~ {depth_end} m  Rows: {rows}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(window, disp)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            draw_state()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, img)
    cv2.setMouseCallback(window, on_click)
    draw_state()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC: 重置
            corners.clear()
            draw_state()
        elif key == 13 and len(corners) == 4:  # ENTER: 确认
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    calib = {
        "image": os.path.basename(image_path),
        "corners": corners,
        "depth_start": depth_start,
        "depth_end": depth_end,
        "rows": rows,
    }

    json_path = os.path.splitext(image_path)[0] + "_calib.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    print(f"标定已保存: {json_path}")
    print(f"  四角: {corners}")
    print(f"  深度: {depth_start} ~ {depth_end} m, {rows} 行")


# ---------------------------------------------------------------------------
# 推理模式
# ---------------------------------------------------------------------------

def load_calibration(image_path):
    """查找并加载与图像对应的标定 JSON"""
    json_path = os.path.splitext(image_path)[0] + "_calib.json"
    if not os.path.exists(json_path):
        print(f"未找到标定文件: {json_path}")
        print("请先运行: python core_depth_mapper.py --mode calibrate --image ...")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pixel_to_depth(px, py, calib):
    """将像素坐标 (px, py) 映射为深度值 (多行同向布局)"""
    corners = np.float32(calib["corners"])
    rows = calib["rows"]
    d_start = calib["depth_start"]
    d_end = calib["depth_end"]

    # 透视变换: 源四角 → 目标矩形
    src = corners  # TL, TR, BR, BL
    # 用源四角的平均边长确定目标宽高
    tl, tr, br, bl = corners
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    W = int((top_w + bot_w) / 2)
    H = int((left_h + right_h) / 2)
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])

    M = cv2.getPerspectiveTransform(src, dst)

    # 变换检测点
    pt = np.array([[[px, py]]], dtype=np.float32)
    pt_rect = cv2.perspectiveTransform(pt, M)
    rx, ry = pt_rect[0][0]

    # 多行同向: 每行左→右深度递增
    row_h = H / rows
    row_idx = int(ry // row_h)
    row_idx = max(0, min(row_idx, rows - 1))
    col_ratio = rx / W
    col_ratio = max(0.0, min(col_ratio, 1.0))

    depth_per_row = (d_end - d_start) / rows
    depth = d_start + (row_idx + col_ratio) * depth_per_row
    return depth


def detect(image_path, model_path, conf=0.25, output=None):
    """运行 YOLO 检测并输出带深度的结果"""
    if not HAS_YOLO:
        print("未安装 ultralytics，请先: pip install ultralytics")
        sys.exit(1)

    calib = load_calibration(image_path)

    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    # 加载模型并推理
    print(f"加载 YOLO 模型: {model_path}")
    model = YOLO(model_path)
    results = model(img, conf=conf, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        names = model.names if hasattr(model, "names") else {}

        for bbox, conf_val, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2  # 用检测框底部偏下的位置作为深度参考点
            depth = pixel_to_depth(cx, cy, calib)
            cls_name = names.get(cls_id, str(cls_id))

            detections.append({
                "class": cls_name,
                "class_id": int(cls_id),
                "confidence": round(float(conf_val), 4),
                "depth": round(float(depth), 3),
                "bbox": [round(float(v), 1) for v in bbox],
            })

    # 按深度排序
    detections.sort(key=lambda d: d["depth"])

    # 输出
    if output is None:
        output = os.path.splitext(image_path)[0] + "_depth.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print(f"检测到 {len(detections)} 个目标，结果已保存: {output}")
    for d in detections:
        print(f"  {d['class']:12s}  conf={d['confidence']:.2f}  depth={d['depth']:.3f}m  "
              f"bbox={d['bbox']}")

    return detections


# ---------------------------------------------------------------------------
# 可视化模式 (可选: 在照片上画出检测框+深度)
# ---------------------------------------------------------------------------

def visualize(image_path, detections_json=None, rectified=False):
    """在照片上叠加检测结果和深度值。rectified=True 时输出透视校正后的矩形图"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    if detections_json is None:
        detections_json = os.path.splitext(image_path)[0] + "_depth.json"
    with open(detections_json, "r", encoding="utf-8") as f:
        detections = json.load(f)

    calib = load_calibration(image_path)
    corners = np.float32(calib["corners"])  # TL, TR, BR, BL

    colors = {
        "crack": (0, 0, 255),
        "crushed zone": (255, 0, 0),
        "intrusion": (255, 0, 0),
        "lithologic variation": (0, 255, 0),
    }

    if rectified:
        # 计算透视变换矩阵
        tl, tr, br, bl = corners
        top_w = np.linalg.norm(tr - tl)
        bot_w = np.linalg.norm(br - bl)
        left_h = np.linalg.norm(bl - tl)
        right_h = np.linalg.norm(br - tr)
        W = int((top_w + bot_w) / 2)
        H = int((left_h + right_h) / 2)
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        M = cv2.getPerspectiveTransform(corners, dst)

        # 校正图像
        img = cv2.warpPerspective(img, M, (W, H))

        # 校正检测框坐标：将 bbox 四角分别做透视变换后取外接矩形
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            corners_box = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(corners_box, M)[0]
            rx1 = min(warped[:, 0]); ry1 = min(warped[:, 1])
            rx2 = max(warped[:, 0]); ry2 = max(warped[:, 1])
            d["bbox_rect"] = [rx1, ry1, rx2, ry2]

        for d in detections:
            x1, y1, x2, y2 = map(int, d.get("bbox_rect", d["bbox"]))
            cls_name = d.get("class", "")
            color = colors.get(cls_name, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {d['depth']:.2f}m"
            cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        suffix = "_rectified.jpg"
    else:
        # 画标定框
        pts = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 255, 0), 2)

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cls_name = d.get("class", "")
            color = colors.get(cls_name, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {d['depth']:.2f}m"
            cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        suffix = "_annotated.jpg"

    vis_path = os.path.splitext(image_path)[0] + suffix
    cv2.imwrite(vis_path, img)
    print(f"可视化结果已保存: {vis_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="岩芯照片深度映射工具")
    parser.add_argument("--mode", required=True,
                        choices=["calibrate", "detect", "visualize"])
    parser.add_argument("--image", required=True, help="照片路径")
    parser.add_argument("--depth", nargs=2, type=float, metavar=("START", "END"),
                        help="深度区间 (米), 仅 calibrate 模式需要")
    parser.add_argument("--rows", type=int, help="岩芯箱行数, 仅 calibrate 模式需要")
    parser.add_argument("--model", help="YOLO 权重路径 (.pt), 仅 detect 模式需要")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--output", help="输出 JSON 路径, 默认与照片同名")
    parser.add_argument("--detections", help="检测结果 JSON, visualize 模式用")
    parser.add_argument("--rectified", action="store_true", help="visualize 模式下输出透视校正后的矩形图")

    args = parser.parse_args()

    if args.mode == "calibrate":
        if args.depth is None or args.rows is None:
            parser.error("calibrate 模式需要 --depth 和 --rows")
        calibrate(args.image, args.depth[0], args.depth[1], args.rows)

    elif args.mode == "detect":
        if args.model is None:
            parser.error("detect 模式需要 --model")
        detect(args.image, args.model, args.conf, args.output)

    elif args.mode == "visualize":
        visualize(args.image, args.detections, getattr(args, 'rectified', False))


if __name__ == "__main__":
    main()
