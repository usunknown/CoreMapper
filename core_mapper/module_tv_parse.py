"""
钻孔电视文件名解析模块
"""
import re
import os


def parse_tv_filename(filename):
    """
    解析电视图像文件名，提取 钻孔号 和 深度区间。

    格式：{borehole}_{params}_{date}_{time}_{interval_cm}_{start_cm}.jpg

    例: "26_762_75_0_20250731_140702_250_750.jpg"
      → borehole_id="26", interval_cm=250, start_cm=750
      → z_top=7.50 m, z_bottom=10.00 m

    返回 None 如果格式不匹配。
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    parts = name.split('_')

    if len(parts) < 8:
        return None

    borehole_id = parts[0]

    # 倒数两个字段是 interval_cm 和 start_cm
    try:
        start_cm_raw = parts[-1]
        interval_cm_raw = parts[-2]
        interval_cm = int(interval_cm_raw)
        start_cm = int(start_cm_raw)
    except (ValueError, IndexError):
        return None

    z_top = start_cm / 100.0
    z_bottom = z_top + interval_cm / 100.0

    return {
        "borehole_id": borehole_id,
        "z_top": z_top,
        "z_bottom": z_bottom,
        "interval_cm": interval_cm,
        "start_cm": start_cm,
    }
