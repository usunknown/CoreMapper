"""
Module 1: 梯形校正 — 四角标定 + 透视变换
"""
import json
import os
import cv2
import numpy as np


def calibrate_interactive(image_path):
    """
    交互式标定：点击 TL-TR-BR-BL 四个角。
    ENTER 确认，ESC 重置，Q 退出。
    返回 corners [(x,y),...] 或 None（用户放弃）。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    corners = []
    title = "CoreMapper Calibrate - [TL TR BR BL]  ENTER=ok  ESC=reset  Q=quit"
    disp = img.copy()

    def draw():
        nonlocal disp
        disp = img.copy()
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(disp, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(disp, ["TL", "TR", "BR", "BL"][i],
                        (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(corners) == 4:
            pts = np.array(corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
        cv2.imshow(title, disp)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            draw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(title, on_click)
    draw()

    result = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            corners.clear()
            draw()
        elif key == 13 and len(corners) == 4:  # ENTER
            result = corners.copy()
            break
        elif key == ord('q'):
            result = None
            break

    cv2.destroyAllWindows()
    return result


def load_calibration(image_path):
    """加载已有标定 JSON"""
    json_path = os.path.splitext(image_path)[0] + "_calib.json"
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_calibration(image_path, corners, depth_start=None, depth_end=None, rows=None):
    """保存标定到 JSON"""
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
    return json_path


def rectify(image_path, calib=None):
    """
    执行透视校正，返回校正后的图像和变换矩阵。
    如果 calib 为 None，尝试从 _calib.json 加载。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if calib is None:
        calib = load_calibration(image_path)
    if calib is None:
        raise ValueError(f"未找到标定数据: {image_path}")

    corners = np.float32(calib["corners"])
    tl, tr, br, bl = corners
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    W = int((top_w + bot_w) / 2)
    H = int((left_h + right_h) / 2)
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(corners, dst)
    rectified = cv2.warpPerspective(img, M, (W, H))
    return rectified, M, (W, H)


def rectify_all(image_dir, progress_callback=None):
    """
    批量校正目录下所有 JPG，跳过无标定的图片。
    progress_callback(current, total) 返回 True 表示取消。
    """
    jpgs = sorted([f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg'))])
    done, skipped = 0, 0

    for jpg in jpgs:
        path = os.path.join(image_dir, jpg)
        calib = load_calibration(path)
        if calib is None:
            skipped += 1
            continue
        try:
            rect, _, _ = rectify(path, calib)
            out_path = os.path.splitext(path)[0] + "_rectified.jpg"
            cv2.imwrite(out_path, rect)
            done += 1
        except Exception as e:
            print(f"  Error {jpg}: {e}")
        if progress_callback:
            if progress_callback(done, len(jpgs)):
                break

    return done, skipped
