from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QFileDialog,
    QScrollArea,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont
import qtawesome as qta


class InferencePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # 创建顶部工具栏
        toolbar = self.create_toolbar()
        layout.addLayout(toolbar)

        # 创建主要内容区域
        content = QHBoxLayout()

        # 左侧图片预览区域
        preview_frame = self.create_preview_frame()
        content.addWidget(preview_frame, stretch=1)

        # 右侧结果展示区域
        result_frame = self.create_result_frame()
        content.addWidget(result_frame, stretch=1)

        layout.addLayout(content)

        # 创建底部状态栏
        status_bar = self.create_status_bar()
        layout.addWidget(status_bar)

    def create_toolbar(self):
        toolbar = QHBoxLayout()

        # 模型选择按钮
        model_btn = QPushButton("选择模型")
        model_btn.setIcon(qta.icon("fa5s.cube"))
        model_btn.setIconSize(QSize(24, 24))
        toolbar.addWidget(model_btn)

        # 图片上传按钮
        upload_btn = QPushButton("上传图片")
        upload_btn.setIcon(qta.icon("fa5s.upload"))
        upload_btn.setIconSize(QSize(24, 24))
        upload_btn.clicked.connect(self.upload_image)
        toolbar.addWidget(upload_btn)

        # 摄像头按钮
        camera_btn = QPushButton("打开摄像头")
        camera_btn.setIcon(qta.icon("fa5s.camera"))
        camera_btn.setIconSize(QSize(24, 24))
        toolbar.addWidget(camera_btn)

        # 批量处理按钮
        batch_btn = QPushButton("批量处理")
        batch_btn.setIcon(qta.icon("fa5s.folder-open"))
        batch_btn.setIconSize(QSize(24, 24))
        toolbar.addWidget(batch_btn)

        toolbar.addStretch()

        return toolbar

    def create_preview_frame(self):
        frame = QFrame()
        frame.setStyleSheet(
            """
            QFrame {
                background-color: #3b3b3b;
                border-radius: 8px;
                padding: 16px;
            }
        """
        )

        layout = QVBoxLayout(frame)

        # 预览标题
        title = QLabel("图片预览")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 预览区域
        preview_area = QScrollArea()
        preview_area.setWidgetResizable(True)
        preview_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
                border-radius: 4px;
            }
        """
        )

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet(
            """
            QLabel {
                background-color: #2b2b2b;
                border-radius: 4px;
            }
        """
        )

        preview_area.setWidget(self.preview_label)
        layout.addWidget(preview_area)

        return frame

    def create_result_frame(self):
        frame = QFrame()
        frame.setStyleSheet(
            """
            QFrame {
                background-color: #3b3b3b;
                border-radius: 8px;
                padding: 16px;
            }
        """
        )

        layout = QVBoxLayout(frame)

        # 结果标题
        title = QLabel("分类结果")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 主要结果
        self.result_label = QLabel("等待图片...")
        self.result_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        # 置信度
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 16))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #4b4b4b;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """
        )
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # 详细信息
        details_frame = QFrame()
        details_frame.setStyleSheet(
            """
            QFrame {
                background-color: #2b2b2b;
                border-radius: 4px;
                padding: 8px;
            }
        """
        )
        details_layout = QVBoxLayout(details_frame)

        details_title = QLabel("详细信息")
        details_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        details_layout.addWidget(details_title)

        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)

        layout.addWidget(details_frame)

        return frame

    def create_status_bar(self):
        frame = QFrame()
        frame.setStyleSheet(
            """
            QFrame {
                background-color: #3b3b3b;
                border-radius: 4px;
                padding: 8px;
            }
        """
        )

        layout = QHBoxLayout(frame)

        # 状态信息
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        # 处理时间
        self.time_label = QLabel("")
        layout.addWidget(self.time_label)

        layout.addStretch()

        return frame

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            # 显示图片
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled_pixmap)

            # 更新状态
            self.status_label.setText("正在处理...")
            self.progress_bar.show()
            self.progress_bar.setValue(0)

            # 模拟推理过程
            self.simulate_inference()

    def simulate_inference(self):
        # 这里应该替换为实际的推理代码
        import time
        import random

        # 模拟处理时间
        for i in range(101):
            time.sleep(0.02)
            self.progress_bar.setValue(i)

        # 模拟结果
        classes = ["玉露", "生石花", "熊童子", "吉娃娃", "虹之玉"]
        result = random.choice(classes)
        confidence = random.uniform(0.85, 0.99)

        self.result_label.setText(result)
        self.confidence_label.setText(f"置信度: {confidence:.2%}")
        self.details_label.setText(
            f"类别: {result}\n"
            f"置信度: {confidence:.2%}\n"
            f"处理时间: 2.1秒\n"
            f"模型版本: v1.0.0"
        )

        self.status_label.setText("处理完成")
        self.time_label.setText("处理时间: 2.1秒")
