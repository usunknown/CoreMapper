"""
钻孔电视图像区域标定模块
"""
import json
import os

import cv2
import numpy as np


CALIB_FILENAME = "tv_calib.json"


def load_tv_calib(tv_dir):
    """加载已有标定文件，不存在则返回 None"""
    path = os.path.join(tv_dir, CALIB_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tv_calib(tv_dir, x0, y0, x1, y1, borehole_id=None, note=""):
    """保存 TV 标定到 tv_calib.json"""
    calib = {
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "borehole_id": borehole_id,
        "note": note,
    }
    path = os.path.join(tv_dir, CALIB_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    return path


def calibrate_interactive(tv_dir, image_path=None):
    """
    交互式标定 TV 图像区域。
    - 若 tv_dir 下无标定文件，打开第一张图让用户画矩形。
    - 用户拖矩形：鼠标按下→拖动→释放，ESC 重置，ENTER 确认，Q 退出。
    - 然后用最后一张图验证，OK 则保存，否则放弃。
    返回 calib dict 或 None。
    """
    from .module_tv_parse import parse_tv_filename

    jpgs = sorted([
        f for f in os.listdir(tv_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not jpgs:
        print("No images found in TV directory")
        return None

    if image_path is None:
        image_path = os.path.join(tv_dir, jpgs[0])

    img = _imread_rgb(image_path)
    h, w = img.shape[:2]
    roi = {"x0": 0, "y0": 0, "x1": w - 1, "y1": h - 1}
    drawing = False
    start_pt = None

    title = "TV Calibration - Drag rectangle [ENTER=confirm ESC=reset Q=quit]"
    disp = img.copy()

    def redraw():
        nonlocal disp
        disp = img.copy()
        cv2.rectangle(
            disp,
            (roi["x0"], roi["y0"]),
            (roi["x1"], roi["y1"]),
            (0, 255, 0), 2,
        )
        cv2.putText(
            disp,
            f"TV: {roi['x1']-roi['x0']+1} x {roi['y1']-roi['y0']+1} px",
            (roi["x0"] + 5, roi["y0"] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow(title, disp)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi["x0"], roi["y0"] = min(start_pt[0], x), min(start_pt[1], y)
            roi["x1"], roi["y1"] = max(start_pt[0], x), max(start_pt[1], y)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi["x0"], roi["y0"] = min(start_pt[0], x), min(start_pt[1], y)
            roi["x1"], roi["y1"] = max(start_pt[0], x), max(start_pt[1], y)
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(title, on_mouse)
    redraw()

    result = None
    # Already loaded tv_calib.json check
    existing = load_tv_calib(tv_dir)
    if existing:
        roi["x0"] = existing["x0"]
        roi["y0"] = existing["y0"]
        roi["x1"] = existing["x1"]
        roi["y1"] = existing["y1"]
        redraw()
        print(f"Loaded existing calibration: {existing}")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            if existing:
                roi["x0"] = existing["x0"]
                roi["y0"] = existing["y0"]
                roi["x1"] = existing["x1"]
                roi["y1"] = existing["y1"]
            else:
                roi["x0"], roi["y0"] = 0, 0
                roi["x1"], roi["y1"] = w - 1, h - 1
            redraw()
        elif key == 13:  # ENTER
            result = {
                "x0": roi["x0"], "y0": roi["y0"],
                "x1": roi["x1"], "y1": roi["y1"],
                "width": roi["x1"] - roi["x0"] + 1,
                "height": roi["y1"] - roi["y0"] + 1,
            }
            break
        elif key == ord('q'):
            result = None
            break

    cv2.destroyWindow(title)
    cv2.waitKey(1)  # 确保窗口被销毁

    return result


def crop_tv_image(image_path, calib):
    """按 calib 裁剪 TV 有效区域，返回 numpy BGR 数组"""
    img = _imread_rgb(image_path)
    if img is None:
        return None
    x0, y0, x1, y1 = calib["x0"], calib["y0"], calib["x1"], calib["y1"]
    return img[y0:y1+1, x0:x1+1]


def _imread_rgb(path):
    """PIL 读取 → numpy RGB → cv2 BGR"""
    from PIL import Image
    img_pil = Image.open(path).convert("RGB")
    img_np = np.array(img_pil)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
