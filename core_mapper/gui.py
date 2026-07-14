"""
CoreMapper GUI — 岩芯 + TV 钻孔照片处理工具
"""
import json
import os
import subprocess
import sys
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QDoubleSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QSpinBox,
)
from PySide6.QtCore import Qt, Signal, QObject


class WorkerSignals(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)


# ================================================================
# 主窗口
# ================================================================

class CoreMapperWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoreMapper — 岩芯 + TV 照片处理工具")
        self.resize(950, 700)
        self.signals = WorkerSignals()
        self.signals.log.connect(self._log)
        self.signals.progress.connect(self._progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)
        self._det_models = []   # 岩芯模型
        self._tv_det_models = []  # TV 模型
        self._rev_cache = None
        self._tv_rev_cache = None
        self._build_ui()

    # ================================================================
    # UI 搭建
    # ================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── 钻孔配置区 ──
        cfg = QGroupBox("钻孔配置")
        cl = QHBoxLayout(cfg)
        cl.addWidget(QLabel("岩芯照片:"))
        self.core_dir = QLineEdit()
        self.core_dir.setPlaceholderText("岩芯照片目录...")
        cl.addWidget(self.core_dir, 1)
        btn_b1 = QPushButton("浏览")
        btn_b1.clicked.connect(lambda: self._browse_dir(self.core_dir))
        cl.addWidget(btn_b1)

        cl.addSpacing(12)
        cl.addWidget(QLabel("TV图像:"))
        self.tv_dir = QLineEdit()
        self.tv_dir.setPlaceholderText("钻孔电视图像目录...")
        cl.addWidget(self.tv_dir, 1)
        btn_b2 = QPushButton("浏览")
        btn_b2.clicked.connect(lambda: self._browse_dir(self.tv_dir))
        cl.addWidget(btn_b2)

        layout.addWidget(cfg)

        # ── Tab 区 ──
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_rectify(), "岩芯-梯形校正")
        self.tabs.addTab(self._build_tab_core_detect(), "岩芯-特征识别")
        self.tabs.addTab(self._build_tab_core_review(), "岩芯-审核修正")
        self.tabs.addTab(self._build_tab_tv_calib(), "TV-图像标定")
        self.tabs.addTab(self._build_tab_tv_detect(), "TV-特征识别")
        self.tabs.addTab(self._build_tab_tv_review(), "TV-审核修正")
        self.tabs.addTab(self._build_tab_database(), "建库导出")
        layout.addWidget(self.tabs)

        # ── 底部日志 ──
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(140)
        layout.addWidget(self.log_area)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    # ================================================================
    # Tab 1 — 岩芯梯形校正
    # ================================================================
    def _build_tab_rectify(self):
        w = QWidget(); l = QVBoxLayout(w)
        bl = QHBoxLayout()
        btn_calib = QPushButton("逐张标定四角")
        btn_calib.clicked.connect(self._rectify_calibrate)
        btn_batch = QPushButton("批量校正已有标定")
        btn_batch.clicked.connect(self._rectify_batch)
        bl.addWidget(btn_calib); bl.addWidget(btn_batch)
        l.addLayout(bl); l.addStretch(); return w

    # ================================================================
    # Tab 2 — 岩芯特征识别
    # ================================================================
    def _build_tab_core_detect(self):
        w = QWidget(); l = QVBoxLayout(w)

        ml = QHBoxLayout()
        ml.addWidget(QLabel("模型路径:"))
        self.c_model_path = QLineEdit("D:/code/SAM3/best.pt")
        ml.addWidget(self.c_model_path, 2)
        ml.addWidget(QLabel("类别:"))
        self.c_classes = QLineEdit("crack")
        ml.addWidget(self.c_classes, 1)
        btn_m = QPushButton("添加岩芯模型")
        btn_m.clicked.connect(self._core_add_model)
        ml.addWidget(btn_m)
        l.addLayout(ml)

        self.c_model_list = QTextEdit(); self.c_model_list.setReadOnly(True)
        self.c_model_list.setMaximumHeight(50); l.addWidget(self.c_model_list)

        fl = QHBoxLayout()
        fl.addWidget(QLabel("置信度:"))
        self.c_conf = QDoubleSpinBox(); self.c_conf.setRange(0.01, 1.0)
        self.c_conf.setValue(0.25); self.c_conf.setSingleStep(0.05)
        fl.addWidget(self.c_conf); fl.addStretch(); l.addLayout(fl)

        btn_d = QPushButton("批量推理（岩芯）"); btn_d.clicked.connect(self._core_detect)
        l.addWidget(btn_d); l.addStretch(); return w

    def _core_add_model(self):
        p = self.c_model_path.text(); cs = [x.strip() for x in self.c_classes.text().split(",") if x.strip()]
        if p and cs:
            self._det_models.append({"path": p, "classes": cs})
            self.c_model_list.setText("\n".join(
                f'{m["path"]} → {", ".join(m["classes"])}' for m in self._det_models))

    # ================================================================
    # Tab 3 — 岩芯审核修正
    # ================================================================
    def _build_tab_core_review(self):
        w = QWidget(); l = QVBoxLayout(w)
        bl = QHBoxLayout()
        btn_s = QPushButton("准备审核文件")
        btn_s.clicked.connect(self._core_review_prepare)
        btn_f = QPushButton("刷新审核结果")
        btn_f.clicked.connect(self._core_review_refresh)
        bl.addWidget(btn_s); bl.addWidget(btn_f); bl.addStretch(); l.addLayout(bl)
        self.c_rev_table = QTableWidget(0, 5)
        self.c_rev_table.setHorizontalHeaderLabels(["图片", "特征类", "检测数", "状态", "审核文件"])
        self.c_rev_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.c_rev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.c_rev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.c_rev_table.doubleClicked.connect(lambda idx: self._review_open_labelme(idx, "core"))
        l.addWidget(self.c_rev_table, 1)
        return w

    # ================================================================
    # Tab 4 — TV 图像标定
    # ================================================================
    def _build_tab_tv_calib(self):
        w = QWidget(); l = QVBoxLayout(w)

        g1 = QGroupBox("有效电视图区域 (像素坐标)")
        gl1 = QHBoxLayout(g1)
        gl1.addWidget(QLabel("X 左:")); self.tv_x0 = QSpinBox(); self.tv_x0.setRange(0, 2000); gl1.addWidget(self.tv_x0)
        gl1.addWidget(QLabel("X 右:")); self.tv_x1 = QSpinBox(); self.tv_x1.setRange(0, 2000); gl1.addWidget(self.tv_x1)
        gl1.addWidget(QLabel("Y 上:")); self.tv_y0 = QSpinBox(); self.tv_y0.setRange(0, 10000); gl1.addWidget(self.tv_y0)
        gl1.addWidget(QLabel("Y 下:")); self.tv_y1 = QSpinBox(); self.tv_y1.setRange(0, 10000); gl1.addWidget(self.tv_y1)
        l.addWidget(g1)

        bl = QHBoxLayout()
        btn_save = QPushButton("保存 TV 标定")
        btn_save.clicked.connect(self._tv_save_calib)
        bl.addWidget(btn_save)
        btn_delete = QPushButton("删除已有标定")
        btn_delete.clicked.connect(self._tv_delete_calib)
        bl.addWidget(btn_delete)
        l.addLayout(bl)

        # 自动加载已有标定
        self.tv_calib_status = QLabel("未标定")
        l.addWidget(self.tv_calib_status)

        btn_test = QPushButton("测试标定（显示边界叠加到第一张图）")
        btn_test.clicked.connect(self._tv_test_calib)
        l.addWidget(btn_test)

        l.addWidget(QLabel("提示：从已有分析数据可知 CK12 通常为 X=224~575 Y=83~6125"))
        l.addStretch(); return w

    def _tv_refresh_calib_ui(self):
        d = self.tv_dir.text()
        if not d: return
        from .module_tv_calib import load_tv_calib
        calib = load_tv_calib(d)
        if calib:
            self.tv_x0.setValue(calib["x0"]); self.tv_x1.setValue(calib["x1"])
            self.tv_y0.setValue(calib["y0"]); self.tv_y1.setValue(calib["y1"])
            self.tv_calib_status.setText(
                f"已标定: x={calib['x0']}~{calib['x1']} ({calib['x1']-calib['x0']+1}px)  "
                f"y={calib['y0']}~{calib['y1']} ({calib['y1']-calib['y0']+1}px)")
        else:
            self.tv_calib_status.setText("未标定")

    def _tv_save_calib(self):
        d = self.tv_dir.text()
        if not d: return
        from .module_tv_calib import save_tv_calib
        save_tv_calib(d, self.tv_x0.value(), self.tv_y0.value(),
                      self.tv_x1.value(), self.tv_y1.value())
        self._tv_refresh_calib_ui()
        self._log("TV 标定已保存: tv_calib.json")

    def _tv_delete_calib(self):
        d = self.tv_dir.text()
        if not d: return
        import os
        path = os.path.join(d, "tv_calib.json")
        if os.path.exists(path):
            os.remove(path)
            self.tv_calib_status.setText("未标定")
            self._log("TV 标定已删除")
        else:
            self._log("无标定文件可删除")

    def _tv_test_calib(self):
        d = self.tv_dir.text()
        if not d: return
        from .module_tv_calib import load_tv_calib, crop_tv_image
        import cv2
        calib = load_tv_calib(d)
        if not calib:
            self._log("请先保存 TV 标定"); return
        jpgs = sorted([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        if not jpgs: return
        # 在第一张图上画矩形框 + 保存
        img = cv2.imread(os.path.join(d, jpgs[0]))
        if img is None: return
        cv2.rectangle(img, (calib["x0"], calib["y0"]), (calib["x1"], calib["y1"]), (0,255,0), 2)
        out_path = os.path.join(d, "_tv_calib_check.jpg")
        cv2.imwrite(out_path, img)
        self._log(f"边界叠加已保存: {out_path}")

    def _tv_calibrate(self):
        # 打开 Tab 4 后自动刷新已有标定显示
        self._tv_refresh_calib_ui()

    # ================================================================
    # Tab 5 — TV 特征识别
    # ================================================================
    def _build_tab_tv_detect(self):
        w = QWidget(); l = QVBoxLayout(w)
        ml = QHBoxLayout()
        ml.addWidget(QLabel("TV模型:"))
        self.tv_model_path = QLineEdit("D:/code/SAM3/best.pt")
        ml.addWidget(self.tv_model_path, 2)
        ml.addWidget(QLabel("类别:"))
        self.tv_classes = QLineEdit("fracture")
        ml.addWidget(self.tv_classes, 1)
        btn_m = QPushButton("添加TV模型")
        btn_m.clicked.connect(self._tv_add_model)
        ml.addWidget(btn_m); l.addLayout(ml)

        self.tv_model_list = QTextEdit(); self.tv_model_list.setReadOnly(True)
        self.tv_model_list.setMaximumHeight(50); l.addWidget(self.tv_model_list)

        fl = QHBoxLayout()
        fl.addWidget(QLabel("置信度:"))
        self.tv_conf = QDoubleSpinBox(); self.tv_conf.setRange(0.01, 1.0)
        self.tv_conf.setValue(0.25); self.tv_conf.setSingleStep(0.05)
        fl.addWidget(self.tv_conf); fl.addStretch(); l.addLayout(fl)

        btn_d = QPushButton("批量推理（TV）"); btn_d.clicked.connect(self._tv_detect)
        l.addWidget(btn_d); l.addStretch(); return w

    def _tv_add_model(self):
        p = self.tv_model_path.text(); cs = [x.strip() for x in self.tv_classes.text().split(",") if x.strip()]
        if p and cs:
            self._tv_det_models.append({"path": p, "classes": cs})
            self.tv_model_list.setText("\n".join(
                f'{m["path"]} → {", ".join(m["classes"])}' for m in self._tv_det_models))

    # ================================================================
    # Tab 6 — TV 审核修正
    # ================================================================
    def _build_tab_tv_review(self):
        w = QWidget(); l = QVBoxLayout(w)
        bl = QHBoxLayout()
        btn_s = QPushButton("准备TV审核文件")
        btn_s.clicked.connect(self._tv_review_prepare)
        btn_f = QPushButton("刷新TV审核结果")
        btn_f.clicked.connect(self._tv_review_refresh)
        bl.addWidget(btn_s); bl.addWidget(btn_f); bl.addStretch(); l.addLayout(bl)
        self.tv_rev_table = QTableWidget(0, 5)
        self.tv_rev_table.setHorizontalHeaderLabels(["图片", "特征类", "检测数", "状态", "审核文件"])
        self.tv_rev_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tv_rev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tv_rev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tv_rev_table.doubleClicked.connect(lambda idx: self._review_open_labelme(idx, "tv"))
        l.addWidget(self.tv_rev_table, 1)
        return w

    # ================================================================
    # Tab 7 — 建库导出
    # ================================================================
    def _build_tab_database(self):
        w = QWidget(); l = QVBoxLayout(w)
        btn = QPushButton("导出 CSV + JSON（含岩芯+TV）")
        btn.clicked.connect(self._export_database)
        l.addWidget(btn); l.addStretch(); return w

    # ================================================================
    # 通用工具
    # ================================================================
    def _log(self, msg): self.log_area.append(msg)
    def _progress(self, cur, tot):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(tot); self.progress_bar.setValue(cur)
    def _on_finished(self): self.progress_bar.setVisible(False); self._log("--- 完成 ---")
    def _on_error(self, msg): QMessageBox.critical(self, "错误", msg)
    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d: le.setText(d)

    def _run_worker(self, target, *args):
        def w():
            try: target(*args)
            except Exception as e: self.signals.error.emit(str(e))
            finally: self.signals.finished.emit()
        threading.Thread(target=w, daemon=True).start()

    # ================================================================
    # 岩芯业务逻辑（复用已有模块）
    # ================================================================
    def _rectify_calibrate(self):
        d = self.core_dir.text()
        if not d: return
        self._run_worker(self._do_calibrate, d)

    def _do_calibrate(self, d):
        from .module_rectify import calibrate_interactive, save_calibration, load_calibration
        from .module_database import parse_filename
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.jpg','.jpeg')): continue
            path = os.path.join(d, f)
            if load_calibration(path): continue
            self.signals.log.emit(f"[标定] {f}")
            corners = calibrate_interactive(path)
            if corners is None: self.signals.log.emit("用户中断标定"); return
            info = parse_filename(f)
            save_calibration(path, corners,
                             info["depth_start"] if info else 0,
                             info["depth_end"] if info else 1,
                             info["rows"] if info else 1)

    def _rectify_batch(self):
        d = self.core_dir.text()
        if not d: return
        self._run_worker(self._do_rectify_batch, d)

    def _do_rectify_batch(self, d):
        from .module_rectify import rectify_all
        def cb(cur, tot): self.signals.progress.emit(cur, tot); return False
        done, skipped = rectify_all(d, cb)
        self.signals.log.emit(f"校正完成: {done} 张, 跳过 {skipped} 张")

    def _core_detect(self):
        d = self.core_dir.text()
        if not d: return
        self._run_worker(self._do_core_detect, d, self.c_conf.value())

    def _do_core_detect(self, d, conf):
        from .module_detect import detect_on_directory
        def cb(cur, tot): self.signals.progress.emit(cur, tot); return False
        n = detect_on_directory(d, self._det_models, conf, cb)
        self.signals.log.emit(f"岩芯识别完成: {n} 个特征")

    # ── 审核（共用） ──
    def _core_review_prepare(self):
        d = self.core_dir.text()
        if not d: return
        self._run_worker(lambda: self._do_review_prepare(d, self.c_rev_table, "core"))

    def _core_review_refresh(self):
        d = self.core_dir.text()
        if not d: return
        self._run_worker(lambda: self._do_review_refresh(d, self.c_rev_table, "core"))

    def _tv_review_prepare(self):
        d = self.tv_dir.text()
        if not d: return
        self._run_worker(lambda: self._do_review_prepare(d, self.tv_rev_table, "tv"))

    def _tv_review_refresh(self):
        d = self.tv_dir.text()
        if not d: return
        self._run_worker(lambda: self._do_review_refresh(d, self.tv_rev_table, "tv"))

    def _do_review_prepare(self, d, table, mode):
        from .module_review import export_all_for_review
        n = export_all_for_review(d)
        self.signals.log.emit(f"已生成 {n} 个审核文件")
        self._populate_review_table(d, table, mode)

    def _do_review_refresh(self, d, table, mode):
        from .module_review import import_all_reviewed
        n = import_all_reviewed(d)
        self.signals.log.emit(f"刷新完成: {n} 张图的检测结果已更新")
        self._populate_review_table(d, table, mode)

    def _populate_review_table(self, d, table, mode):
        import glob
        review_base = os.path.join(d, "review")
        review_files = sorted(glob.glob(os.path.join(review_base, "*", "*_review.json")))
        if not review_files:
            review_files = sorted(glob.glob(os.path.join(d, "*_review.json")))

        table.setRowCount(len(review_files))
        for i, rp in enumerate(review_files):
            base = os.path.basename(rp).replace("_review.json", "")
            cls_name = os.path.basename(os.path.dirname(rp))
            try:
                with open(rp, encoding="utf-8") as f: data = json.load(f)
                n = len(data.get("shapes", []))
            except: n = 0; cls_name = "?"
            table.setItem(i, 0, QTableWidgetItem(base))
            table.setItem(i, 1, QTableWidgetItem(cls_name))
            table.setItem(i, 2, QTableWidgetItem(str(n)))
            table.setItem(i, 3, QTableWidgetItem("待审核"))
            table.setItem(i, 4, QTableWidgetItem(os.path.basename(rp)))

        if mode == "core":
            self._rev_cache = d
        else:
            self._tv_rev_cache = d

    def _review_open_labelme(self, index, mode):
        d = self._rev_cache if mode == "core" else self._tv_rev_cache
        table = self.c_rev_table if mode == "core" else self.tv_rev_table
        if not d: return
        row = index.row()
        fname = table.item(row, 4).text()
        cls_name = table.item(row, 1).text()
        base = fname.replace("_review.json", "")
        review_dir = os.path.join(d, "review", cls_name)
        img = os.path.join(review_dir, base + "_review.jpg")
        rp = os.path.join(review_dir, fname)
        if not os.path.exists(rp):
            rp = os.path.join(d, fname)
            img = os.path.join(d, base + "_review.jpg")
        if not os.path.exists(rp): return
        if not os.path.exists(img): img = os.path.join(d, base + ".jpg")
        self.signals.log.emit(f"启动 labelme: {cls_name}/{base}")
        lm = sys.executable.replace("python.exe", "Scripts/labelme.exe")
        try:
            if os.path.exists(lm): subprocess.Popen([lm, img])
            else: subprocess.Popen([sys.executable, "-m", "labelme", img])
        except FileNotFoundError:
            subprocess.Popen([sys.executable, "-m", "labelme", img])

    # ================================================================
    # TV 标定
    # ================================================================
    def _tv_calibrate(self):
        d = self.tv_dir.text()
        if not d: return
        from .module_tv_calib import calibrate_interactive, save_tv_calib
        result = calibrate_interactive(d)
        if result:
            save_tv_calib(d, result["x0"], result["y0"], result["x1"], result["y1"])
            self.tv_calib_status.setText(
                f"已标定: x={result['x0']}~{result['x1']} ({result['width']}px)  "
                f"y={result['y0']}~{result['y1']} ({result['height']}px)")
            self._log(f"TV 标定已保存: tv_calib.json")
        else:
            self._log("TV 标定已取消")

    # ================================================================
    # TV 推理
    # ================================================================
    def _tv_detect(self):
        d = self.tv_dir.text()
        if not d: return
        self._run_worker(self._do_tv_detect, d, self.tv_conf.value())

    def _do_tv_detect(self, d, conf):
        from .module_detect import detect_tv_directory
        def cb(cur, tot): self.signals.progress.emit(cur, tot); return False
        n = detect_tv_directory(d, self._tv_det_models, conf, cb)
        self.signals.log.emit(f"TV 识别完成: {n} 个特征")

    # ================================================================
    # 建库导出
    # ================================================================
    def _export_database(self):
        core_d = self.core_dir.text()
        tv_d = self.tv_dir.text()
        if not core_d and not tv_d: return
        self._run_worker(self._do_export, core_d, tv_d)

    def _do_export(self, core_d, tv_d):
        from .module_database import collect_detections, export_csv, export_json

        all_records = []
        if core_d:
            self.signals.log.emit("收集岩芯检测结果...")
            cr = collect_detections(core_d)
            for r in cr: r["source"] = "core"
            all_records.extend(cr)
            self.signals.log.emit(f"  岩芯: {len(cr)} 条记录")

        if tv_d:
            self.signals.log.emit("收集 TV 检测结果...")
            tr = collect_detections(tv_d)
            for r in tr: r["source"] = "tv"
            all_records.extend(tr)
            self.signals.log.emit(f"  TV: {len(tr)} 条记录")

        # 统一输出到岩芯目录；如果岩芯目录为空则输出到 TV 目录
        out_dir = core_d or tv_d
        csv_path = os.path.join(out_dir, "feature_database.csv")
        json_path = os.path.join(out_dir, "feature_database.json")
        export_csv(all_records, csv_path)
        export_json(all_records, json_path)
        self.signals.log.emit(f"导出完成: {len(all_records)} 条记录")
        self.signals.log.emit(f"  CSV:  {csv_path}")
        self.signals.log.emit(f"  JSON: {json_path}")


# ================================================================
# 启动
# ================================================================
def run_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CoreMapperWindow()
    window.show()
    sys.exit(app.exec())
