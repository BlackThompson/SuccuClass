import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QFrame,
    QSizePolicy,
    QScrollArea,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor
import qtawesome as qta
import darkdetect
from inference_page import InferencePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SuccuClass - 多肉植物分类系统")
        self.setMinimumSize(1200, 800)

        # 设置主题
        self.is_dark = darkdetect.isDark()
        self.setup_theme()

        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 创建侧边栏
        self.sidebar = self.create_sidebar()
        self.main_layout.addWidget(self.sidebar)

        # 创建内容区域
        self.content = self.create_content_area()
        self.main_layout.addWidget(self.content)

        # 设置初始页面
        self.setup_pages()

    def setup_theme(self):
        if self.is_dark:
            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #3b3b3b;
                    color: #ffffff;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #4b4b4b;
                }
                QLabel {
                    color: #ffffff;
                }
                QFrame {
                    border: 1px solid #3b3b3b;
                }
            """
            )
        else:
            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                QWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                QPushButton {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #dddddd;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #f5f5f5;
                }
                QLabel {
                    color: #000000;
                }
                QFrame {
                    border: 1px solid #dddddd;
                }
            """
            )

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet(
            """
            QFrame {
                border-right: 1px solid #3b3b3b;
            }
        """
        )

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)

        # Logo区域
        logo_frame = QFrame()
        logo_layout = QVBoxLayout(logo_frame)
        logo_label = QLabel("SuccuClass")
        logo_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(logo_label)
        layout.addWidget(logo_frame)

        # 导航按钮
        nav_buttons = [
            ("主页", qta.icon("fa5s.home")),
            ("数据集", qta.icon("fa5s.database")),
            ("训练", qta.icon("fa5s.play")),
            ("模型", qta.icon("fa5s.cube")),
            ("推理", qta.icon("fa5s.search")),
            ("可视化", qta.icon("fa5s.chart-line")),
            ("设置", qta.icon("fa5s.cog")),
        ]

        for text, icon in nav_buttons:
            btn = QPushButton(text)
            btn.setIcon(icon)
            btn.setIconSize(QSize(24, 24))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, t=text: self.change_page(t))
            layout.addWidget(btn)

        layout.addStretch()

        # 主题切换按钮
        theme_btn = QPushButton("切换主题")
        theme_btn.setIcon(qta.icon("fa5s.moon"))
        theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(theme_btn)

        return sidebar

    def create_content_area(self):
        content = QFrame()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建堆叠窗口部件
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        return content

    def setup_pages(self):
        # 创建各个页面
        self.pages = {
            "主页": self.create_home_page(),
            "数据集": self.create_dataset_page(),
            "训练": self.create_training_page(),
            "模型": self.create_model_page(),
            "推理": InferencePage(),
            "可视化": self.create_visualization_page(),
            "设置": self.create_settings_page(),
        }

        # 添加页面到堆叠窗口
        for page in self.pages.values():
            self.stack.addWidget(page)

    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 欢迎信息
        welcome = QLabel("欢迎使用 SuccuClass")
        welcome.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome)

        # 统计信息卡片
        stats_layout = QHBoxLayout()
        stats = [
            ("数据集大小", "1,234 张图片"),
            ("模型数量", "3 个"),
            ("训练次数", "12 次"),
            ("准确率", "95.6%"),
        ]

        for title, value in stats:
            card = QFrame()
            card.setStyleSheet(
                """
                QFrame {
                    background-color: #3b3b3b;
                    border-radius: 8px;
                    padding: 16px;
                }
            """
            )
            card_layout = QVBoxLayout(card)

            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 12))
            value_label = QLabel(value)
            value_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))

            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            stats_layout.addWidget(card)

        layout.addLayout(stats_layout)

        # 最近活动
        recent_frame = QFrame()
        recent_layout = QVBoxLayout(recent_frame)

        recent_title = QLabel("最近活动")
        recent_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        recent_layout.addWidget(recent_title)

        activities = [
            "完成模型训练 - 2024-03-23",
            "添加新数据集 - 2024-03-22",
            "更新模型架构 - 2024-03-21",
            "优化训练参数 - 2024-03-20",
        ]

        for activity in activities:
            activity_label = QLabel(activity)
            recent_layout.addWidget(activity_label)

        layout.addWidget(recent_frame)

        return page

    def create_dataset_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 数据集管理工具栏
        toolbar = QHBoxLayout()

        add_btn = QPushButton("添加数据集")
        add_btn.setIcon(qta.icon("fa5s.plus"))
        toolbar.addWidget(add_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setIcon(qta.icon("fa5s.sync"))
        toolbar.addWidget(refresh_btn)

        layout.addLayout(toolbar)

        # 数据集列表
        dataset_frame = QFrame()
        dataset_layout = QVBoxLayout(dataset_frame)

        datasets = [
            ("训练集", "800 张图片", "2024-03-23"),
            ("验证集", "200 张图片", "2024-03-23"),
            ("测试集", "234 张图片", "2024-03-23"),
        ]

        for name, size, date in datasets:
            dataset_card = QFrame()
            dataset_card.setStyleSheet(
                """
                QFrame {
                    background-color: #3b3b3b;
                    border-radius: 8px;
                    padding: 16px;
                }
            """
            )
            card_layout = QHBoxLayout(dataset_card)

            info_layout = QVBoxLayout()
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            size_label = QLabel(size)
            date_label = QLabel(date)

            info_layout.addWidget(name_label)
            info_layout.addWidget(size_label)
            info_layout.addWidget(date_label)

            card_layout.addLayout(info_layout)
            card_layout.addStretch()

            preview_btn = QPushButton("预览")
            preview_btn.setIcon(qta.icon("fa5s.eye"))
            card_layout.addWidget(preview_btn)

            dataset_layout.addWidget(dataset_card)

        layout.addWidget(dataset_frame)

        return page

    def create_training_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 训练配置
        config_frame = QFrame()
        config_layout = QVBoxLayout(config_frame)

        config_title = QLabel("训练配置")
        config_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        config_layout.addWidget(config_title)

        # 添加配置选项
        configs = [
            ("模型选择", "ResNet50"),
            ("学习率", "0.001"),
            ("批次大小", "32"),
            ("训练轮数", "100"),
            ("数据增强", "启用"),
        ]

        for label, value in configs:
            config_row = QHBoxLayout()
            config_row.addWidget(QLabel(label))
            config_row.addWidget(QLabel(value))
            config_row.addStretch()
            config_layout.addLayout(config_row)

        layout.addWidget(config_frame)

        # 训练控制
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)

        start_btn = QPushButton("开始训练")
        start_btn.setIcon(qta.icon("fa5s.play"))
        control_layout.addWidget(start_btn)

        stop_btn = QPushButton("停止训练")
        stop_btn.setIcon(qta.icon("fa5s.stop"))
        control_layout.addWidget(stop_btn)

        layout.addWidget(control_frame)

        # 训练进度
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)

        progress_title = QLabel("训练进度")
        progress_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        progress_layout.addWidget(progress_title)

        # 添加进度显示
        progress_layout.addWidget(QLabel("当前轮次: 0/100"))
        progress_layout.addWidget(QLabel("当前批次: 0/25"))
        progress_layout.addWidget(QLabel("损失值: 0.000"))
        progress_layout.addWidget(QLabel("准确率: 0.00%"))

        layout.addWidget(progress_frame)

        return page

    def create_model_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 模型列表
        model_frame = QFrame()
        model_layout = QVBoxLayout(model_frame)

        model_title = QLabel("模型列表")
        model_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        model_layout.addWidget(model_title)

        models = [
            ("ResNet50", "95.6%", "2024-03-23"),
            ("EfficientNet", "94.8%", "2024-03-22"),
            ("MobileNetV3", "93.2%", "2024-03-21"),
        ]

        for name, accuracy, date in models:
            model_card = QFrame()
            model_card.setStyleSheet(
                """
                QFrame {
                    background-color: #3b3b3b;
                    border-radius: 8px;
                    padding: 16px;
                }
            """
            )
            card_layout = QHBoxLayout(model_card)

            info_layout = QVBoxLayout()
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            accuracy_label = QLabel(f"准确率: {accuracy}")
            date_label = QLabel(f"训练日期: {date}")

            info_layout.addWidget(name_label)
            info_layout.addWidget(accuracy_label)
            info_layout.addWidget(date_label)

            card_layout.addLayout(info_layout)
            card_layout.addStretch()

            export_btn = QPushButton("导出")
            export_btn.setIcon(qta.icon("fa5s.download"))
            card_layout.addWidget(export_btn)

            model_layout.addWidget(model_card)

        layout.addWidget(model_frame)

        return page

    def create_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 可视化工具栏
        toolbar = QHBoxLayout()

        chart_types = ["损失曲线", "准确率曲线", "混淆矩阵", "特征分布"]
        for chart_type in chart_types:
            btn = QPushButton(chart_type)
            toolbar.addWidget(btn)

        layout.addLayout(toolbar)

        # 图表显示区域
        chart_frame = QFrame()
        chart_frame.setStyleSheet(
            """
            QFrame {
                background-color: #3b3b3b;
                border-radius: 8px;
                padding: 16px;
            }
        """
        )
        chart_layout = QVBoxLayout(chart_frame)

        # 添加图表占位符
        chart_placeholder = QLabel("图表显示区域")
        chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_placeholder.setFont(QFont("Arial", 20))
        chart_layout.addWidget(chart_placeholder)

        layout.addWidget(chart_frame)

        return page

    def create_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 设置选项
        settings = [
            ("主题设置", "深色/浅色"),
            ("语言", "中文/English"),
            ("数据存储路径", "D:/Data/SuccuClass"),
            ("模型保存路径", "D:/Models/SuccuClass"),
            ("自动保存", "启用/禁用"),
            ("GPU加速", "启用/禁用"),
        ]

        for title, value in settings:
            setting_frame = QFrame()
            setting_layout = QHBoxLayout(setting_frame)

            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 12))
            value_label = QLabel(value)

            setting_layout.addWidget(title_label)
            setting_layout.addStretch()
            setting_layout.addWidget(value_label)

            layout.addWidget(setting_frame)

        # 保存按钮
        save_btn = QPushButton("保存设置")
        save_btn.setIcon(qta.icon("fa5s.save"))
        layout.addWidget(save_btn)

        return page

    def change_page(self, page_name):
        # 更新按钮状态
        for btn in self.sidebar.findChildren(QPushButton):
            btn.setChecked(btn.text() == page_name)

        # 切换页面
        self.stack.setCurrentWidget(self.pages[page_name])

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.setup_theme()

        # 更新主题切换按钮图标
        theme_btn = self.sidebar.findChild(QPushButton, "切换主题")
        if theme_btn:
            theme_btn.setIcon(qta.icon("fa5s.moon" if self.is_dark else "fa5s.sun"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
