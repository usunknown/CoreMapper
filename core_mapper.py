"""
CoreMapper — 岩芯照片批量处理与特征建库工具

用法:
    # 启动 GUI
    python core_mapper.py

    # CLI: 批量校正
    python core_mapper.py rectify --dir <图片目录>

    # CLI: 批量识别
    python core_mapper.py detect --dir <图片目录> --model best.pt --classes crack --conf 0.25

    # CLI: 建库导出
    python core_mapper.py export --dir <检测结果目录>
"""
import argparse
import sys


def cmd_rectify(args):
    from core_mapper.module_rectify import rectify_all
    done, skipped = rectify_all(args.dir)
    print(f"校正完成: {done} 张, 跳过 {skipped} 张")


def cmd_detect(args):
    import json
    from core_mapper.module_detect import detect_on_directory
    models = [{"path": args.model, "classes": args.classes.split(",")}]
    results = detect_on_directory(args.dir, models, args.conf)
    total = sum(len(v) for v in results.values())
    print(f"识别完成: {len(results)} 张图, {total} 个特征")


def cmd_export(args):
    from core_mapper.module_database import collect_detections, export_csv
    import os
    records = collect_detections(args.dir)
    output = os.path.join(args.dir, "feature_database.csv")
    export_csv(records, output)
    print(f"导出完成: {len(records)} 条记录 -> {output}")


def main():
    parser = argparse.ArgumentParser(description="CoreMapper")
    sub = parser.add_subparsers(dest="command")

    p_r = sub.add_parser("rectify")
    p_r.add_argument("--dir", required=True)

    p_d = sub.add_parser("detect")
    p_d.add_argument("--dir", required=True)
    p_d.add_argument("--model", required=True)
    p_d.add_argument("--classes", default="crack")
    p_d.add_argument("--conf", type=float, default=0.25)

    p_x = sub.add_parser("export")
    p_x.add_argument("--dir", required=True)

    args = parser.parse_args()

    if args.command == "rectify":
        cmd_rectify(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        # 默认启动 GUI
        from core_mapper.gui import run_gui
        run_gui()


if __name__ == "__main__":
    main()
