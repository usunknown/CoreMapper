"""
CoreMapper GUI — 三 Tab 界面: 梯形校正 / 特征识别 / 建库导出
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
    QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView,
)
from PySide6.QtCore import Qt, Signal, QObject


# ---- 跨线程进度信号 ----
class WorkerSignals(QObject):
    progress = Signal(int, int)  # current, total
    log = Signal(str)
    finished = Signal()
    error = Signal(str)


# ---- 主窗口 ----
class CoreMapperWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoreMapper — 岩芯照片处理工具")
        self.resize(900, 650)
        self.signals = WorkerSignals()
        self.signals.log.connect(self._log)
        self.signals.progress.connect(self._progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)

        self._build_ui()

    # ================================================================
    # UI
    # ================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_rectify(), "梯形校正")
        self.tabs.addTab(self._build_tab_detect(), "特征识别")
        self.tabs.addTab(self._build_tab_review(), "审核修正")
        self.tabs.addTab(self._build_tab_database(), "建库导出")
        layout.addWidget(self.tabs)

        # 底部日志区
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    # ---- Tab 1: 梯形校正 ----
    def _build_tab_rectify(self):
        w = QWidget()
        l = QVBoxLayout(w)

        # 目录选择
        dg = QGroupBox("图片目录")
        dl = QHBoxLayout(dg)
        self.rect_dir = QLineEdit()
        self.rect_dir.setPlaceholderText("选择岩芯照片目录...")
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(lambda: self._browse_dir(self.rect_dir))
        dl.addWidget(self.rect_dir, 1)
        dl.addWidget(btn_browse)
        l.addWidget(dg)

        # 操作按钮
        bl = QHBoxLayout()
        btn_calib = QPushButton("逐张标定四角")
        btn_calib.clicked.connect(self._rectify_calibrate)
        btn_batch = QPushButton("批量校正已有标定")
        btn_batch.clicked.connect(self._rectify_batch)
        bl.addWidget(btn_calib)
        bl.addWidget(btn_batch)
        l.addLayout(bl)

        l.addStretch()
        return w

    # ---- Tab 2: 特征识别 ----
    def _build_tab_detect(self):
        w = QWidget()
        l = QVBoxLayout(w)

        dg = QGroupBox("图片目录与模型配置")
        dl = QVBoxLayout(dg)

        # 目录
        hl = QHBoxLayout()
        self.det_dir = QLineEdit()
        self.det_dir.setPlaceholderText("选择包含 _rectified.jpg 的目录...")
        btn_b = QPushButton("浏览")
        btn_b.clicked.connect(lambda: self._browse_dir(self.det_dir))
        hl.addWidget(self.det_dir, 1)
        hl.addWidget(btn_b)
        dl.addLayout(hl)

        # 模型路径
        md = QHBoxLayout()
        md.addWidget(QLabel("模型:"))
        self.model_path = QLineEdit("D:/code/SAM3/best.pt")
        self.model_classes = QLineEdit("crack")
        md.addWidget(self.model_path, 2)
        md.addWidget(QLabel("类别:"))
        md.addWidget(self.model_classes, 1)
        btn_add_m = QPushButton("添加模型")
        btn_add_m.clicked.connect(self._add_model)
        md.addWidget(btn_add_m)
        dl.addLayout(md)

        # 已添加模型列表
        self.model_list = QTextEdit()
        self.model_list.setReadOnly(True)
        self.model_list.setMaximumHeight(60)
        self.model_list.setPlaceholderText("已添加的模型将显示在此...")
        dl.addWidget(self.model_list)

        # 置信度
        cf = QHBoxLayout()
        cf.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        cf.addWidget(self.conf_spin)
        cf.addStretch()
        dl.addLayout(cf)

        l.addWidget(dg)

        # 操作
        self.det_models = []
        btn_detect = QPushButton("批量推理")
        btn_detect.clicked.connect(self._detect_batch)
        l.addWidget(btn_detect)

        l.addStretch()
        return w

    def _add_model(self):
        path = self.model_path.text()
        classes = [c.strip() for c in self.model_classes.text().split(",") if c.strip()]
        if not path or not classes:
            return
        self.det_models.append({"path": path, "classes": classes})
        # 更新显示
        lines = [f'{m["path"]}  →  {", ".join(m["classes"])}' for m in self.det_models]
        self.model_list.setText("\n".join(lines))
        self._log(f"添加模型: {path} ({classes})")

    # ---- Tab 3: 审核修正 ----
    def _build_tab_review(self):
        w = QWidget()
        l = QVBoxLayout(w)

        dg = QGroupBox("图片目录")
        dl = QHBoxLayout(dg)
        self.rev_dir = QLineEdit()
        self.rev_dir.setPlaceholderText("选择包含 _rectified.jpg 和 _detections.json 的目录...")
        btn_b = QPushButton("浏览")
        btn_b.clicked.connect(lambda: self._browse_dir(self.rev_dir))
        dl.addWidget(self.rev_dir, 1)
        dl.addWidget(btn_b)
        l.addWidget(dg)

        bl = QHBoxLayout()
        btn_scan = QPushButton("准备审核文件")
        btn_scan.clicked.connect(self._review_prepare)
        self.btn_review_refresh = QPushButton("刷新审核结果")
        self.btn_review_refresh.clicked.connect(self._review_refresh)
        bl.addWidget(btn_scan)
        bl.addWidget(self.btn_review_refresh)
        bl.addStretch()
        l.addLayout(bl)

        self.review_table = QTableWidget(0, 4)
        self.review_table.setHorizontalHeaderLabels(["图片", "检测数", "状态", "审核文件"])
        self.review_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.review_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.review_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.review_table.doubleClicked.connect(self._review_open_labelme)
        l.addWidget(self.review_table, 1)

        l.addWidget(QLabel("双击某行 → 自动打开 labelme 审核 → 关闭保存 → 回这里点\"刷新审核结果\""))

        self._rev_dir_cache = None
        return w

    # ---- Tab 4: 建库导出 ----
    def _build_tab_database(self):
        w = QWidget()
        l = QVBoxLayout(w)

        dg = QGroupBox("检测结果目录")
        dl = QHBoxLayout(dg)
        self.db_dir = QLineEdit()
        self.db_dir.setPlaceholderText("选择包含 _detections.json 的目录...")
        btn_b = QPushButton("浏览")
        btn_b.clicked.connect(lambda: self._browse_dir(self.db_dir))
        dl.addWidget(self.db_dir, 1)
        dl.addWidget(btn_b)
        l.addWidget(dg)

        btn_export = QPushButton("导出 CSV + JSON")
        btn_export.clicked.connect(self._export_database)
        l.addWidget(btn_export)

        l.addStretch()
        return w

    # ================================================================
    # 业务逻辑
    # ================================================================
    def _log(self, msg):
        self.log_area.append(msg)

    def _progress(self, cur, total):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)

    def _on_finished(self):
        self.progress_bar.setVisible(False)
        self._log("--- 完成 ---")

    def _on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)

    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            line_edit.setText(d)

    def _run_worker(self, target, *args):
        """在后台线程运行函数，通过 signals 通信"""
        def wrapper():
            try:
                target(*args)
            except Exception as e:
                self.signals.error.emit(str(e))
            finally:
                self.signals.finished.emit()
        threading.Thread(target=wrapper, daemon=True).start()

    # ---- 梯形校正 ----
    def _rectify_calibrate(self):
        d = self.rect_dir.text()
        if not d:
            return
        self._log("启动逐张标定（ESC 跳过已标/不需标的图，Q 退出标定）...")
        self._run_worker(self._do_calibrate, d)

    def _do_calibrate(self, d):
        from .module_rectify import calibrate_interactive, save_calibration, load_calibration
        from .module_database import parse_filename

        jpgs = sorted([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg'))])
        for i, f in enumerate(jpgs):
            path = os.path.join(d, f)
            if load_calibration(path) is not None:
                self.signals.progress.emit(i + 1, len(jpgs))
                continue

            self.signals.log.emit(f"[{i+1}/{len(jpgs)}] {f}")
            corners = calibrate_interactive(path)
            if corners is None:
                self.signals.log.emit("用户中断标定")
                return

            # 从文件名自动解析深度和排数
            info = parse_filename(f)
            if info:
                depth_start = info["depth_start"]
                depth_end = info["depth_end"]
                rows = info["rows"]
            else:
                depth_start = 0.0
                depth_end = 1.0
                rows = 1

            save_calibration(path, corners, depth_start, depth_end, rows)
            self.signals.log.emit(f"  已保存: depth={depth_start}-{depth_end}m rows={rows}")
            self.signals.progress.emit(i + 1, len(jpgs))

    def _rectify_batch(self):
        d = self.rect_dir.text()
        if not d:
            return
        self._run_worker(self._do_rectify_batch, d)

    def _do_rectify_batch(self, d):
        from .module_rectify import rectify_all
        def cb(cur, tot):
            self.signals.progress.emit(cur, tot)
            self.signals.log.emit(f"  已校正 {cur}/{tot}")
            return False
        done, skipped = rectify_all(d, cb)
        self.signals.log.emit(f"校正完成: {done} 张, 跳过 {skipped} 张(无标定)")

    # ---- 特征识别 ----
    def _detect_batch(self):
        d = self.det_dir.text()
        if not d:
            return
        if not self.det_models:
            self._add_model()  # 用当前输入
            if not self.det_models:
                QMessageBox.warning(self, "提示", "请先添加模型")
                return
        conf = self.conf_spin.value()
        self._run_worker(self._do_detect_batch, d, conf)

    def _do_detect_batch(self, d, conf):
        from .module_detect import detect_on_directory
        def cb(cur, tot):
            self.signals.progress.emit(cur, tot)
            self.signals.log.emit(f"  已处理 {cur}/{tot}")
            return False
        results = detect_on_directory(d, self.det_models, conf, cb)
        total_dets = sum(len(v) for v in results.values())
        self.signals.log.emit(f"识别完成: {len(results)} 张图, {total_dets} 个特征")

    # ---- 审核修正 Tab 方法 ----
    def _review_prepare(self):
        """为目录下所有已检测图生成 _review.json"""
        d = self.rev_dir.text() or self.det_dir.text()
        if not d:
            return
        self._run_worker(self._do_review_prepare, d)

    def _do_review_prepare(self, d):
        from .module_review import export_all_for_review
        n = export_all_for_review(d)
        self.signals.log.emit(f"已生成 {n} 个 _review.json")
        self._populate_review_table(d)

    def _review_refresh(self):
        d = self.rev_dir.text() or self.det_dir.text()
        if not d:
            return
        self._run_worker(self._do_review_refresh, d)

    def _do_review_refresh(self, d):
        from .module_review import import_all_reviewed
        n = import_all_reviewed(d)
        self.signals.log.emit(f"刷新完成: {n} 张图的检测结果已更新（覆盖 _detections.json）")
        self._populate_review_table(d)

    def _populate_review_table(self, d):
        import glob
        self._rev_dir_cache = d
        review_files = sorted(
            glob.glob(os.path.join(d, "*_review.json"))
            + glob.glob(os.path.join(d, "*_rectified.json"))
        )
        self.review_table.setRowCount(len(review_files))
        for i, rp in enumerate(review_files):
            base = os.path.basename(rp).replace("_review.json", "")
            # 检查是否有对应原图
            jpg = base + ".jpg" if os.path.exists(os.path.join(d, base + ".jpg")) else (base + ".JPG" if os.path.exists(os.path.join(d, base + ".JPG")) else "")
            # 读检测数
            try:
                with open(rp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n = len(data.get("shapes", []))
                reviewed = all(s.get("flags", {}).get("reviewed", False) for s in data.get("shapes", []))
                status = "已审核" if reviewed else "待审核"
            except:
                n = 0
                status = "?"
            self.review_table.setItem(i, 0, QTableWidgetItem(base))
            self.review_table.setItem(i, 1, QTableWidgetItem(str(n)))
            self.review_table.setItem(i, 2, QTableWidgetItem(status))
            self.review_table.setItem(i, 3, QTableWidgetItem(os.path.basename(rp)))
        self.signals.log.emit(f"表格刷新: {len(review_files)} 个文件")

    def _review_open_labelme(self, index):
        d = self._rev_dir_cache or self.rev_dir.text()
        if not d:
            return
        row = index.row()
        fname = self.review_table.item(row, 3).text()
        rp = os.path.join(d, fname)
        if not os.path.exists(rp):
            self.signals.log.emit(f"文件不存在: {rp}")
            return
        # 找到对应的校正图
        base = fname.replace("_review.json", "")
        rect = os.path.join(d, base + "_rectified.jpg")
        if not os.path.exists(rect):
            rect = os.path.join(d, base + ".jpg")
        self.signals.log.emit(f"启动 labelme: {base}")
        labelme = sys.executable.replace("python.exe", "Scripts/labelme.exe")
        try:
            if os.path.exists(labelme):
                subprocess.Popen([labelme, rect])
            else:
                subprocess.Popen([sys.executable, "-m", "labelme", rect])
        except FileNotFoundError:
            subprocess.Popen([sys.executable, "-m", "labelme", rect])

    # ---- 建库导出 ----
    def _export_database(self):
        d = self.db_dir.text()
        if not d:
            return
        self._run_worker(self._do_export, d)

    def _do_export(self, d):
        from .module_database import collect_detections, export_csv, export_json
        self.signals.log.emit("收集检测结果...")
        records = collect_detections(d)
        self.signals.log.emit(f"共 {len(records)} 条记录")

        csv_path = os.path.join(d, "feature_database.csv")
        json_path = os.path.join(d, "feature_database.json")
        export_csv(records, csv_path)
        export_json(records, json_path)
        self.signals.log.emit(f"CSV 已导出: {csv_path}")
        self.signals.log.emit(f"JSON 已导出: {json_path}")


# ---- 启动 ----
def run_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CoreMapperWindow()
    window.show()
    sys.exit(app.exec())
