from PySide6.QtCore import Qt, QThread, QSize
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence
from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QSpinBox,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QGroupBox, QFormLayout,
    QCheckBox, QSizePolicy
)

from src.ui.video_worker import VideoWorker


class ClickableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._last_image_size = None  # (w, h)
        self._scaled_pix_size = None  # (w, h)

    def setPixmap(self, pm: QPixmap) -> None:
        super().setPixmap(pm)
        self._scaled_pix_size = (pm.width(), pm.height())

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        mw = self.window()
        if not hasattr(mw, "on_video_click"):
            return super().mousePressEvent(ev)
        mw.on_video_click(ev.position().x(), ev.position().y())
        return super().mousePressEvent(ev)


class MainWindow(QMainWindow):
    src_size = None  # (w, h)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic ROI Vision Demo (YOLOv8)")
        self.resize(1200, 700)

        self.video_label = ClickableLabel("No video started")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#bbb; border:1px solid #333;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(320, 240)

        # Controls
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Mode 1: Full fixed frame",
            "Mode 2: Fixed sub-ROI",
            "Mode 3: Movable ROI",
            "Mode 4: Movable frame + Movable ROI",
        ])
        self.mode_combo.setCurrentIndex(0)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "Video File"])
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 10)
        self.cam_index.setValue(0)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.on_browse)
        self.video_path = None

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(50)  # 0.50
        self.conf_slider.valueChanged.connect(self.on_threshold_changed)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)  # 0.45
        self.iou_slider.valueChanged.connect(self.on_threshold_changed)

        self.autofocus_cb = QCheckBox("Autofocus target (for movable modes)")
        self.autofocus_cb.setChecked(True)
        self.autofocus_cb.stateChanged.connect(self.on_autofocus_changed)

        self.info_label = QLabel(
            "Humans-only detection | CUDA preferred | yolov8n.pt in project root | Click on video to center ROI"
        )
        self.info_label.setStyleSheet("color:#888;")

        # Layouts
        controls_box = QGroupBox("Controls")
        form = QFormLayout()
        form.addRow("Mode", self.mode_combo)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        source_row = QHBoxLayout()
        source_row.addWidget(self.source_combo)
        source_row.addWidget(QLabel("Cam Index"))
        source_row.addWidget(self.cam_index)
        source_row.addWidget(self.browse_btn)
        form.addRow("Source", source_row)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Conf"))
        thresh_row.addWidget(self.conf_slider)
        thresh_row.addSpacing(12)
        thresh_row.addWidget(QLabel("IoU"))
        thresh_row.addWidget(self.iou_slider)
        form.addRow("Thresholds", thresh_row)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.start_btn)
        buttons_row.addWidget(self.stop_btn)
        form.addRow("Run", buttons_row)

        form.addRow(self.autofocus_cb)
        form.addRow(self.info_label)
        controls_box.setLayout(form)

        root = QWidget()
        hl = QHBoxLayout(root)
        hl.addWidget(self.video_label, stretch=3)
        hl.addWidget(controls_box, stretch=2)
        self.setCentralWidget(root)

        # Shortcuts 1-4 to switch modes
        for i in range(4):
            sc = QShortcut(QKeySequence(str(i + 1)), self)
            sc.activated.connect(lambda idx=i: self.mode_combo.setCurrentIndex(idx))

        # Worker Thread
        self.thread: Optional[QThread] = None
        self.worker: Optional[VideoWorker] = None

        # Placeholder image
        self._show_placeholder()

    def _show_placeholder(self):
        import numpy as np
        import cv2
        h, w = 540, 960
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "", (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        cv2.putText(img, "Click Start to run webcam/video", (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 140), 2)
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
        self._set_image(qimg)

    def _set_image(self, qimg: QImage):
        pm = QPixmap.fromImage(qimg)
        # Scale to fit label keeping aspect ratio
        target = self.video_label.size()
        if target.width() < 10 or target.height() < 10:
            self.video_label.setPixmap(pm)
            return
        pm2 = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pm2)
        # Use true source size if known for click mapping
        if self.src_size is not None:
            self.video_label._last_image_size = self.src_size
        else:
            self.video_label._last_image_size = (qimg.width(), qimg.height())
        self.video_label._scaled_pix_size = (pm2.width(), pm2.height())

    def resizeEvent(self, ev):
        # Avoid rescaling pixmap during resize to prevent geometry feedback loops.
        # The image will be scaled on the next frame or when explicitly set.
        return super().resizeEvent(ev)

    def on_browse(self):
        if self.source_combo.currentText() == "Video File":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select video file", filter="Video Files (*.mp4 *.avi *.mkv);;All Files (*.*)"
            )
            if path:
                self.video_path = path
                self.info_label.setText(f"Video selected: {path}")

    def on_start(self):
        # Create worker and thread if needed
        if self.thread is not None:
            return
        self.thread = QThread(self)
        self.worker = VideoWorker()
        self.worker.moveToThread(self.thread)

        # Connections
        self.thread.started.connect(self.worker.run)
        self.worker.frameReady.connect(self._set_image)
        self.worker.metricsReady.connect(lambda s: self.info_label.setText(s))
        self.worker.status.connect(lambda s: self.info_label.setText(s))

        # Forward size mapping
        self.worker.sizeChanged.connect(self.on_size_changed)

        # Configure settings
        source_type = "video" if self.source_combo.currentText() == "Video File" else "webcam"
        if source_type == "video" and not self.video_path:
            self.info_label.setText("Please select a video file.")
            self.thread = None
            self.worker = None
            return

        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        mode = self.mode_combo.currentIndex() + 1
        autofocus = self.autofocus_cb.isChecked()

        self.worker.configure(
            source_type=source_type,
            cam_index=self.cam_index.value(),
            video_path=self.video_path,
            conf=conf,
            iou=iou,
            mode=mode,
            autofocus=autofocus,
            prefer_cuda=True,
        )

        # Start
        self.thread.start()

    def on_stop(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
            self.worker = None
        self.info_label.setText("Stopped")

    def on_mode_changed(self, idx: int):
        if self.worker:
            self.worker.set_mode(idx + 1)

    def on_threshold_changed(self):
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        if self.worker:
            self.worker.set_conf(conf)
            self.worker.set_iou(iou)

    def on_autofocus_changed(self):
        if self.worker:
            self.worker.set_autofocus(self.autofocus_cb.isChecked())

    def on_video_click(self, x: float, y: float):
        # Map click in label coords -> frame coords
        pix = self.video_label.pixmap()
        if pix is None or self.worker is None:
            return
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        scaled_w, scaled_h = self.video_label._scaled_pix_size or (pix.width(), pix.height())
        off_x = (label_w - scaled_w) * 0.5
        off_y = (label_h - scaled_h) * 0.5
        ix = (x - off_x)
        iy = (y - off_y)
        if ix < 0 or iy < 0 or ix >= scaled_w or iy >= scaled_h:
            return
        # Back to original (true) image size
        if self.src_size is not None:
            img_w, img_h = self.src_size
        else:
            img_w, img_h = self.video_label._last_image_size or (pix.width(), pix.height())
        fx = img_w / max(1.0, scaled_w)
        fy = img_h / max(1.0, scaled_h)
        cx = int(ix * fx)
        cy = int(iy * fy)
        self.worker.set_roi_center(cx, cy)

    def on_size_changed(self, w: int, h: int):
        self.src_size = (w, h)
        # Trigger a re-fit of the last pixmap for correct click mapping
        pix = self.video_label.pixmap()
        if pix is not None:
            self._set_image(pix.toImage())

    def closeEvent(self, ev):
        self.on_stop()
        return super().closeEvent(ev)
