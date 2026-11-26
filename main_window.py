"""主窗口：菜单、工具栏、类别配置、图片列表与标注保存。"""

from html import escape as html_escape
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from canvas import SEG_COLORS, AnnotationCanvas, color_for_label, contrast_text_hex
from io_utils import (
    list_images,
    load_polygons_for_image,
    mask_path_for,
    read_boxes_from_xml,
    read_image_size,
    read_polygons_from_xml,
    save_annotation,
    write_mask_png,
)
from models import VERSION, Annotation, Config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"标注工具 {VERSION} - 目标检测(XML) + 语义分割(PNG)")
        self.resize(1440, 900)

        self.config = Config()
        self.annotation = Annotation()
        self.image_files = []
        self.folder_root: Path = None
        self.current_index = -1

        self.canvas = AnnotationCanvas()
        self.setCentralWidget(self.canvas)
        self.canvas.annotationChanged.connect(self.on_annotation_changed)
        self.canvas.annotationChanged.connect(self.auto_save)
        self.canvas.selectionChanged.connect(self.on_selection_changed)
        self.canvas.prevImageRequested.connect(self.prev_image)
        self.canvas.nextImageRequested.connect(self.next_image)
        self.canvas.cursorMoved.connect(self.on_cursor_moved)
        self.canvas.statusHint.connect(self.on_status_hint)

        self._build_menus()
        self._build_toolbar()
        self._build_docks()
        self._build_statusbar()
        self.sync_from_config()
        self.on_annotation_changed()
        self.on_status_hint(
            f"配置加载失败，已保留当前类别且不会覆盖文件: {self.config.last_error}"
            if self.config.last_error
            else "检测模式：拖拽鼠标画框；切换分割模式可绘制多边形"
        )

        self.config_timer = QTimer(self)
        self.config_timer.timeout.connect(self.reload_config_if_changed)
        self.config_timer.start(1000)

    def reload_config_if_changed(self) -> None:
        """配置文件被外部修改后，自动同步到软件界面与画布。"""
        if not self.config.changed_externally():
            return
        self.reload_config_now()

    def reload_config_if_changed_forced(self) -> None:
        self.reload_config_now()

    def reload_config_now(self) -> None:
        if not self.config.load():
            self.on_status_hint(f"配置未更新: {self.config.last_error}")
            return
        self.sync_from_config()
        self.on_status_hint(f"已从 {self.config.path.name} 重新加载类别配置")

    def open_config_file(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config.path)))
        self.on_status_hint(f"配置文件: {self.config.path}（保存后约 1 秒自动同步）")

    def sync_from_config(self) -> None:
        boxes = self.config.box_classes
        self.box_class_edit.blockSignals(True)
        self.box_class_edit.setText(", ".join(boxes))
        self.box_class_edit.blockSignals(False)

        self.combo_box_class.blockSignals(True)
        self.combo_box_class.clear()
        self.combo_box_class.addItems(boxes)
        self.combo_box_class.blockSignals(False)
        if boxes:
            self.combo_box_class.setCurrentIndex(0)
        self.canvas.box_label = self.combo_box_class.currentText() or "object"

        self.selected_label_combo.blockSignals(True)
        self.selected_label_combo.clear()
        self.selected_label_combo.addItems(boxes)
        self.selected_label_combo.blockSignals(False)
        self.on_selection_changed(self.canvas.selected)

        for i, edit in enumerate(self.seg_edits):
            edit.blockSignals(True)
            edit.setText(self.config.seg_classes[i])
            edit.blockSignals(False)
        self.apply_seg_classes(show_message=False, save=False)
        self.refresh_box_legend()

        self.auto_mask_check.blockSignals(True)
        self.auto_mask_check.setChecked(self.config.auto_generate_mask)
        self.auto_mask_check.blockSignals(False)
        suffix = self.config.mask_suffix
        self.suffix_combo.blockSignals(True)
        if suffix in ("", "_mask"):
            self.suffix_combo.setCurrentIndex(1 if suffix else 0)
            self.custom_suffix_edit.setText("")
        else:
            self.suffix_combo.setCurrentIndex(2)
            self.custom_suffix_edit.setText(suffix)
        self.suffix_combo.blockSignals(False)
        self.on_suffix_mode_changed()
        self.apply_output_settings(show_message=False, save=False)

        self.on_annotation_changed()

    # ---------- 界面构建 ----------

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        self._add_action(file_menu, "打开图片", "Ctrl+O", self.open_image)
        self._add_action(file_menu, "打开文件夹", "Ctrl+Shift+O", self.open_folder)
        file_menu.addSeparator()
        self._add_action(file_menu, "保存标注", "Ctrl+S", self.save)
        self._add_action(file_menu, "导出检测 XML", "", self.export_xml)
        self._add_action(file_menu, "导出分割 PNG", "", self.export_mask)
        self._add_action(file_menu, "一键导出所有分割图", "", self.export_all_masks)
        file_menu.addSeparator()
        self._add_action(file_menu, "退出", "Ctrl+Q", self.close)

        edit_menu = menubar.addMenu("编辑")
        self._add_action(edit_menu, "删除选中", "Del", self.delete_selected)
        self._add_action(edit_menu, "清空当前图片标注", "", self.clear_annotations)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "上一张 (↑)", "Ctrl+Left", self.prev_image)
        self._add_action(edit_menu, "下一张 (↓)", "Ctrl+Right", self.next_image)

        view_menu = menubar.addMenu("视图")
        self._add_action(view_menu, "放大", "Ctrl++", self.zoom_in)
        self._add_action(view_menu, "缩小", "Ctrl+-", self.zoom_out)
        self._add_action(view_menu, "适应窗口", "F", self.fit_image)

        settings_menu = menubar.addMenu("设置")
        self.auto_save_action = QAction("自动保存标注（修改后即时写入）", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.setChecked(True)
        settings_menu.addAction(self.auto_save_action)
        settings_menu.addSeparator()
        self._add_action(settings_menu, "编辑配置文件 config.json", "", self.open_config_file)
        self._add_action(settings_menu, "重新加载配置文件", "", self.reload_config_if_changed_forced)
        self._add_action(settings_menu, "选择输出目录", "", self.choose_output_dir)
        self._add_action(settings_menu, "输出目录：图片所在目录", "", self.reset_output_dir)

        help_menu = menubar.addMenu("帮助")
        self._add_action(help_menu, "使用说明", "", self.show_help)
        self._add_action(help_menu, f"关于 ({VERSION})", "", self.show_about)

    def _add_action(self, menu, text: str, shortcut: str, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ApplicationShortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction("打开图片", self.open_image)
        toolbar.addAction("打开文件夹", self.open_folder)
        toolbar.addAction("保存", self.save)
        toolbar.addAction("导出全部Mask", self.export_all_masks)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("模式:"))
        self.btn_detect = QPushButton("检测框")
        self.btn_detect.setCheckable(True)
        self.btn_detect.setChecked(True)
        self.btn_segment = QPushButton("分割区域")
        self.btn_segment.setCheckable(True)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.btn_detect, 0)
        mode_group.addButton(self.btn_segment, 1)
        mode_group.idClicked.connect(self.set_mode)
        toolbar.addWidget(self.btn_detect)
        toolbar.addWidget(self.btn_segment)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("检测类别:"))
        self.combo_box_class = QComboBox()
        self.combo_box_class.addItems(self.config.box_classes)
        self.combo_box_class.currentTextChanged.connect(self.on_box_class_changed)
        toolbar.addWidget(self.combo_box_class)

        toolbar.addWidget(QLabel("分割类别:"))
        self.seg_buttons = []
        self.seg_group = QButtonGroup(self)
        self.seg_group.setExclusive(True)
        for i in range(Config.MAX_SEG_CLASSES):
            button = QPushButton()
            button.setCheckable(True)
            button.setChecked(i == 0)
            self.seg_group.addButton(button, i)
            toolbar.addWidget(button)
            self.seg_buttons.append(button)
        self.seg_group.idClicked.connect(self.on_seg_class_changed)
        toolbar.addSeparator()

        toolbar.addAction("放大", self.zoom_in)
        toolbar.addAction("缩小", self.zoom_out)
        toolbar.addAction("适应窗口", self.fit_image)
        toolbar.addAction("上一张", self.prev_image)
        toolbar.addAction("下一张", self.next_image)

    def _build_docks(self) -> None:
        image_dock = QDockWidget("图片列表", self)
        image_dock.setObjectName("imageDock")
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.itemClicked.connect(self.on_image_list_clicked)
        image_dock.setWidget(self.image_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, image_dock)

        annotation_dock = QDockWidget("标注面板", self)
        annotation_dock.setObjectName("annotationDock")
        tabs = QTabWidget()

        annotation_page = QWidget()
        ann_layout = QVBoxLayout(annotation_page)
        label_row = QHBoxLayout()
        self.selected_label_caption = QLabel("选中类别:")
        label_row.addWidget(self.selected_label_caption)
        self.selected_label_combo = QComboBox()
        self.selected_label_combo.setEditable(False)
        self.selected_label_combo.addItems(self.config.box_classes)
        self.selected_label_combo.setEnabled(False)
        self.selected_label_combo.currentTextChanged.connect(self.on_selected_label_changed)
        label_row.addWidget(self.selected_label_combo)
        ann_layout.addLayout(label_row)
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.on_annotation_item_clicked)
        ann_layout.addWidget(self.annotation_list)
        button_row = QHBoxLayout()
        delete_button = QPushButton("删除选中")
        delete_button.clicked.connect(self.delete_selected)
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self.clear_annotations)
        button_row.addWidget(delete_button)
        button_row.addWidget(clear_button)
        ann_layout.addLayout(button_row)
        tabs.addTab(annotation_page, "标注列表")

        config_page = QWidget()
        config_layout = QVBoxLayout(config_page)

        box_group = QGroupBox("检测类别（写入 XML 的 name 字段）")
        box_layout = QVBoxLayout(box_group)
        self.box_class_edit = QLineEdit(", ".join(self.config.box_classes))
        box_layout.addWidget(QLabel("用逗号分隔多个类别："))
        box_layout.addWidget(self.box_class_edit)
        self.box_color_legend = QLabel()
        self.box_color_legend.setWordWrap(True)
        box_layout.addWidget(self.box_color_legend)
        apply_box_button = QPushButton("应用检测类别")
        apply_box_button.clicked.connect(self.apply_box_classes)
        box_layout.addWidget(apply_box_button)
        config_layout.addWidget(box_group)

        seg_group = QGroupBox("分割类别（最多 3 类，分别写入 R / G / B 通道）")
        seg_layout = QFormLayout(seg_group)
        self.seg_edits = []
        for i, channel in enumerate("RGB"):
            edit = QLineEdit(self.config.seg_classes[i])
            swatch = QLabel("  ")
            swatch.setStyleSheet(
                f"background-color: {SEG_COLORS[i].name()}; color: white; font-weight: bold;"
            )
            swatch.setAlignment(Qt.AlignCenter)
            seg_layout.addRow(f"{channel} 通道 - 类别 {i + 1}", edit)
            self.seg_edits.append(edit)
        apply_seg_button = QPushButton("应用分割类别")
        apply_seg_button.clicked.connect(lambda: self.apply_seg_classes())
        seg_layout.addRow(apply_seg_button)
        config_layout.addWidget(seg_group)

        output_group = QGroupBox("分割图输出设置")
        output_layout = QFormLayout(output_group)
        self.auto_mask_check = QCheckBox("标完分割后自动生成 PNG")
        self.auto_mask_check.setChecked(self.config.auto_generate_mask)
        self.auto_mask_check.setToolTip("关闭后自动保存不再生成 PNG，手动“导出分割 PNG”仍可用")
        output_layout.addRow(self.auto_mask_check)
        self.suffix_combo = QComboBox()
        self.suffix_combo.addItem("使用原名（无后缀）", "")
        self.suffix_combo.addItem("_mask（默认）", "_mask")
        self.suffix_combo.addItem("自定义后缀", "__custom__")
        self.custom_suffix_edit = QLineEdit(self.config.mask_suffix)
        self.custom_suffix_edit.setPlaceholderText("例如 _seg")
        self.suffix_combo.currentIndexChanged.connect(self.on_suffix_mode_changed)
        output_layout.addRow("文件名后缀:", self.suffix_combo)
        output_layout.addRow("自定义:", self.custom_suffix_edit)
        apply_output_button = QPushButton("应用输出设置")
        apply_output_button.clicked.connect(self.apply_output_settings)
        output_layout.addRow(apply_output_button)
        config_layout.addWidget(output_group)
        self.on_suffix_mode_changed()

        config_layout.addStretch()
        tabs.addTab(config_page, "类别配置")

        annotation_dock.setWidget(tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, annotation_dock)

    def _build_statusbar(self) -> None:
        status = self.statusBar()
        self.image_status = QLabel("未打开图片")
        self.mode_status = QLabel("模式: 检测框")
        self.hint_status = QLabel("")
        self.save_status = QLabel("自动保存: 就绪")
        self.position_status = QLabel("x: -, y: -")
        self.version_status = QLabel(VERSION)
        self.version_status.setToolTip("当前软件版本")
        status.addWidget(self.image_status)
        status.addWidget(self.mode_status)
        status.addPermanentWidget(self.hint_status, 1)
        status.addPermanentWidget(self.save_status)
        status.addPermanentWidget(self.position_status)
        status.addPermanentWidget(self.version_status)

    # ---------- 文件操作 ----------

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if not path:
            return
        self.image_files = [Path(path)]
        self.folder_root = self.image_files[0].parent
        self.image_list.clear()
        self.image_list.addItem(Path(path).name)
        self.load_image_at(0)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.folder_root = Path(folder)
        self.image_files = list_images(self.folder_root, recursive=True)
        self.image_list.clear()
        for path in self.image_files:
            try:
                label = str(path.relative_to(self.folder_root))
            except ValueError:
                label = path.name
            self.image_list.addItem(label)
        if self.image_files:
            self.load_image_at(0)
        else:
            QMessageBox.information(self, "提示", "所选文件夹（含子目录）中没有支持的图片")

    def load_image_at(self, index: int) -> None:
        if not 0 <= index < len(self.image_files):
            return
        path = self.image_files[index]
        try:
            width, height = read_image_size(path)
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", f"无法读取 {path.name}: {exc}")
            return

        xml_path = path.with_suffix(".xml")
        existing_boxes = read_boxes_from_xml(xml_path)
        existing_polygons = load_polygons_for_image(
            path, self.config.output_dir, self.config.mask_suffix
        )
        self.current_index = index
        self.annotation = Annotation(image_path=path, width=width, height=height)
        self.annotation.boxes.extend(existing_boxes)
        self.annotation.polygons.extend(existing_polygons)
        self.canvas.box_label = self.combo_box_class.currentText() or "object"
        self.canvas.seg_class = 0
        self.canvas.set_annotation(self.annotation)
        try:
            display_name = str(path.relative_to(self.folder_root)) if self.folder_root else path.name
        except (TypeError, ValueError):
            display_name = path.name
        self.image_status.setText(f"{display_name}  ({width} × {height})")
        self.save_status.setText("自动保存: 就绪")
        self.save_status.setStyleSheet("")
        self.sync_image_list()
        self.on_annotation_changed()
        self.on_status_hint(
            f"已加载 {display_name}，共 {len(self.image_files)} 张图片"
            f"（{len(existing_boxes)} 个目标框，{len(existing_polygons)} 个分割区域）"
        )

    def sync_image_list(self) -> None:
        self.image_list.blockSignals(True)
        self.image_list.clearSelection()
        if 0 <= self.current_index < self.image_list.count():
            self.image_list.setCurrentRow(self.current_index)
            self.image_list.scrollToItem(self.image_list.currentItem())
        self.image_list.blockSignals(False)

    def on_image_list_clicked(self, item: QListWidgetItem) -> None:
        self.load_image_at(self.image_list.row(item))

    def next_image(self) -> None:
        self.load_image_at(self.current_index + 1)

    def prev_image(self) -> None:
        self.load_image_at(self.current_index - 1)

    def save(self) -> None:
        self._do_save(force_xml=False, force_mask=False)

    def export_xml(self) -> None:
        self._do_save(force_xml=True, force_mask=False)

    def export_mask(self) -> None:
        self._do_save(force_xml=False, force_mask=True)

    def export_all_masks(self) -> None:
        """批量导出所有含分割标注的图片掩码，无分割标注的图片跳过。"""
        if not self.image_files:
            QMessageBox.information(self, "提示", "请先打开图片或文件夹")
            return

        total = len(self.image_files)
        progress = QProgressDialog("正在导出分割图...", "取消", 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)
        generated = skipped = failed = 0

        for index, path in enumerate(self.image_files):
            progress.setValue(index)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                if index == self.current_index:
                    polygons = self.annotation.polygons
                else:
                    polygons = load_polygons_for_image(
                        path, self.config.output_dir, self.config.mask_suffix
                    )
                if not polygons:
                    skipped += 1
                    continue
                width, height = read_image_size(path)
                mask_path = mask_path_for(path, self.config.output_dir, self.config.mask_suffix)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                write_mask_png(
                    Annotation(image_path=path, width=width, height=height, polygons=polygons),
                    mask_path,
                )
                generated += 1
            except Exception as exc:
                failed += 1
                self.on_status_hint(f"导出 {path.name} 失败: {exc}")

        progress.setValue(total)
        progress.close()
        QMessageBox.information(
            self,
            "导出完成",
            f"共 {total} 张图片：生成 {generated} 张，跳过 {skipped} 张（无分割标注）"
            + (f"，失败 {failed} 张" if failed else "")
            + f"。\n输出后缀: {self.config.mask_suffix or '（原名）'}",
        )

    def _do_save(self, force_xml: bool, force_mask: bool) -> None:
        if not self.annotation.image_path:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        try:
            xml_path, mask_path = save_annotation(
                self.annotation,
                self.config.output_dir,
                force_xml=force_xml,
                force_mask=force_mask,
                seg_labels=self.config.seg_classes,
                auto_generate_mask=True,
                mask_suffix=self.config.mask_suffix,
            )
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        saved = []
        if xml_path:
            saved.append(f"检测框 XML: {xml_path}")
        if mask_path:
            saved.append(f"分割 PNG:  {mask_path}")
        if saved:
            QMessageBox.information(self, "保存成功", "\n".join(saved))
        else:
            QMessageBox.information(
                self,
                "提示",
                "当前没有可保存的标注。\n检测框生成 .xml，分割区域生成 PNG 掩码。",
            )

    def auto_save(self) -> None:
        """标注发生变化后即时写入文件；已存在的文件即使清空标注也会同步更新。"""
        if not self.auto_save_action.isChecked() or not self.annotation.image_path:
            return
        stem = self.annotation.image_path.stem
        out_dir = self.config.output_dir or self.annotation.image_path.parent
        mask_path = mask_path_for(self.annotation.image_path, out_dir, self.config.mask_suffix)
        try:
            xml_path, mask_path = save_annotation(
                self.annotation,
                self.config.output_dir,
                force_xml=(out_dir / f"{stem}.xml").exists(),
                force_mask=mask_path.exists(),
                seg_labels=self.config.seg_classes,
                auto_generate_mask=self.config.auto_generate_mask,
                mask_suffix=self.config.mask_suffix,
            )
        except Exception as exc:
            self.save_status.setText(f"自动保存失败: {exc}")
            self.save_status.setStyleSheet("color: #dc2626;")
            return
        saved = [name for name, path in (("XML", xml_path), ("PNG", mask_path)) if path]
        self.save_status.setText(
            f"已自动保存 {' + '.join(saved)}" if saved else "标注已清空（无文件变更）"
        )
        self.save_status.setStyleSheet("color: #16a34a;")

    def choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择标注输出目录")
        if folder:
            self.config.output_dir = Path(folder)
            self.config.save()
            self.on_status_hint(f"输出目录: {folder}")

    def reset_output_dir(self) -> None:
        self.config.output_dir = None
        self.config.save()
        self.on_status_hint("输出目录: 图片所在目录")

    # ---------- 类别与模式 ----------

    def set_mode(self, mode_id: int) -> None:
        mode = "detect" if mode_id == 0 else "segment"
        self.canvas.set_mode(mode)
        self.mode_status.setText(f"模式: {'检测框' if mode == 'detect' else '分割区域'}")
        self.combo_box_class.setEnabled(mode == "detect")
        for button in self.seg_buttons:
            button.setEnabled(mode == "segment")

    def on_box_class_changed(self, text: str) -> None:
        self.canvas.box_label = text or "object"

    def on_seg_class_changed(self, class_id: int) -> None:
        self.canvas.seg_class = class_id
        selection = self.canvas.selected
        if selection and selection[0] == "poly" and selection[1] < len(self.annotation.polygons):
            poly = self.annotation.polygons[selection[1]]
            if poly.class_id != class_id:
                poly.class_id = class_id
                self.canvas.redraw()
                self.on_annotation_changed()
                self.auto_save()
        self.on_status_hint(
            f"分割类别: {self.config.seg_classes[class_id]}（写入 {'RGB'[class_id]} 通道）"
        )

    def refresh_box_legend(self) -> None:
        chips = " ".join(
            f'<span style="background-color:{color_for_label(name).name()};'
            f'color:{contrast_text_hex(color_for_label(name))};'
            f'padding:2px 6px;border-radius:3px;">{html_escape(name)}</span>'
            for name in self.config.box_classes
        )
        self.box_color_legend.setText(f"框颜色：{chips}")

    def apply_box_classes(self) -> None:
        raw = self.box_class_edit.text().replace("，", ",")
        names = [name.strip() for name in raw.split(",") if name.strip()]
        if not names:
            QMessageBox.warning(self, "提示", "请至少填写一个检测类别")
            return
        self.config.box_classes = names
        self.config.save()
        self.combo_box_class.blockSignals(True)
        self.combo_box_class.clear()
        self.combo_box_class.addItems(names)
        self.combo_box_class.blockSignals(False)
        self.combo_box_class.setCurrentIndex(0)
        self.canvas.box_label = names[0]
        self.selected_label_combo.blockSignals(True)
        self.selected_label_combo.clear()
        self.selected_label_combo.addItems(names)
        self.selected_label_combo.blockSignals(False)
        self.on_selection_changed(self.canvas.selected)
        self.refresh_box_legend()
        QMessageBox.information(self, "完成", f"已设置 {len(names)} 个检测类别")

    def apply_seg_classes(self, show_message: bool = True, save: bool = True) -> None:
        names = []
        for i in range(Config.MAX_SEG_CLASSES):
            name = self.seg_edits[i].text().strip() if hasattr(self, "seg_edits") else self.config.seg_classes[i]
            names.append(name or f"类别{'RGB'[i]}")
        self.config.seg_classes = names
        if save:
            self.config.save()
        self.canvas.set_seg_class_names(names)
        for i, button in enumerate(self.seg_buttons):
            button.setText(f"{i + 1}: {names[i]}")
            hex_color = SEG_COLORS[i].name()
            button.setStyleSheet(
                f"QPushButton:checked {{ background-color: {hex_color}; "
                f"color: white; font-weight: bold; }}"
            )
        if show_message:
            QMessageBox.information(self, "完成", "已设置分割类别名称")

    def on_suffix_mode_changed(self) -> None:
        mode = self.suffix_combo.currentData()
        self.custom_suffix_edit.setEnabled(mode == "__custom__")

    def apply_output_settings(self, show_message: bool = True, save: bool = True) -> None:
        self.config.auto_generate_mask = self.auto_mask_check.isChecked()
        mode = self.suffix_combo.currentData()
        if mode == "__custom__":
            self.config.mask_suffix = self.custom_suffix_edit.text().strip()
        else:
            self.config.mask_suffix = mode or ""
        if save:
            self.config.save()
        self.on_status_hint(
            f"分割图输出: {'自动生成' if self.config.auto_generate_mask else '不自动生成'}，"
            f"后缀: {self.config.mask_suffix or '（原名）'}"
        )
        if show_message:
            QMessageBox.information(self, "完成", "已应用分割图输出设置")

    # ---------- 标注列表与状态 ----------

    def on_annotation_changed(self) -> None:
        self.annotation_list.clear()
        for i, box in enumerate(self.annotation.boxes):
            b = box.ordered()
            item = QListWidgetItem(
                f"框 {i + 1}: {box.label}  "
                f"[{int(b.xmin)}, {int(b.ymin)} - {int(b.xmax)}, {int(b.ymax)}]"
            )
            item.setData(Qt.UserRole, ("box", i))
            self.annotation_list.addItem(item)
        for i, poly in enumerate(self.annotation.polygons):
            item = QListWidgetItem(
                f"分割 {i + 1}: {self.config.seg_classes[poly.class_id]} "
                f"({len(poly.points)} 个顶点)"
            )
            item.setData(Qt.UserRole, ("poly", i))
            self.annotation_list.addItem(item)
        self.on_selection_changed(self.canvas.selected)

    def on_selected_label_changed(self, text: str) -> None:
        text = text.strip()
        selection = self.canvas.selected
        if not text or not selection:
            return
        if selection[0] == "box":
            if selection[1] >= len(self.annotation.boxes):
                return
            box = self.annotation.boxes[selection[1]]
            if box.label == text:
                return
            box.label = text
            self.canvas.redraw()
            self.on_annotation_changed()
            self.auto_save()
            self.on_status_hint(f"已将框 {selection[1] + 1} 类别修改为 {text}")
        elif selection[0] == "poly":
            if text not in self.config.seg_classes:
                return
            if selection[1] >= len(self.annotation.polygons):
                return
            class_id = self.config.seg_classes.index(text)
            poly = self.annotation.polygons[selection[1]]
            if poly.class_id == class_id:
                return
            poly.class_id = class_id
            self.canvas.seg_class = class_id
            self.seg_buttons[class_id].blockSignals(True)
            self.seg_buttons[class_id].setChecked(True)
            self.seg_buttons[class_id].blockSignals(False)
            self.canvas.redraw()
            self.on_annotation_changed()
            self.auto_save()
            self.on_status_hint(f"已将分割 {selection[1] + 1} 类别修改为 {text}")

    def on_selection_changed(self, selection) -> None:
        self.annotation_list.blockSignals(True)
        self.annotation_list.clearSelection()
        if selection:
            for i in range(self.annotation_list.count()):
                item = self.annotation_list.item(i)
                if item.data(Qt.UserRole) == selection:
                    item.setSelected(True)
                    self.annotation_list.scrollToItem(item)
                    break
        self.annotation_list.blockSignals(False)

        combo = self.selected_label_combo
        combo.blockSignals(True)
        combo.clear()
        if selection and selection[0] == "box" and selection[1] < len(self.annotation.boxes):
            self.selected_label_caption.setText("选中框类别:")
            combo.addItems(self.config.box_classes)
            label = self.annotation.boxes[selection[1]].label
            if combo.findText(label) < 0:
                combo.addItem(label)
            combo.setCurrentText(label)
            combo.setEnabled(True)
        elif selection and selection[0] == "poly" and selection[1] < len(self.annotation.polygons):
            self.selected_label_caption.setText("分割类别:")
            combo.addItems(self.config.seg_classes)
            combo.setCurrentIndex(self.annotation.polygons[selection[1]].class_id)
            combo.setEnabled(True)
        else:
            self.selected_label_caption.setText("选中类别:")
            combo.addItems(self.config.box_classes)
            combo.setEnabled(False)
            combo.setCurrentText("")
        combo.blockSignals(False)

    def on_annotation_item_clicked(self, item: QListWidgetItem) -> None:
        self.canvas.select(item.data(Qt.UserRole))

    def on_cursor_moved(self, x: float, y: float) -> None:
        self.position_status.setText(f"x: {int(x)}, y: {int(y)}")

    def on_status_hint(self, text: str) -> None:
        self.hint_status.setText(text)

    def delete_selected(self) -> None:
        self.canvas.delete_selected()

    def clear_annotations(self) -> None:
        if self.annotation.is_empty():
            return
        reply = QMessageBox.question(
            self, "确认", "确定清空当前图片的所有标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.canvas.clear_annotations()

    # ---------- 视图与帮助 ----------

    def zoom_in(self) -> None:
        self.canvas.zoom(1.2)

    def zoom_out(self) -> None:
        self.canvas.zoom(1 / 1.2)

    def fit_image(self) -> None:
        self.canvas.fit_image()

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "使用说明",
            "目标框（检测模式）\n"
            "  · 拖拽鼠标绘制矩形框，框的类别取自工具栏“检测类别”\n"
            "  · 不同类别的框用不同颜色显示，颜色对应见右侧“类别配置”\n"
            "  · 点击框可选中并整体移动，拖动四角可缩放，Del 删除\n"
            "  · 选中标注后用右侧“选中类别”下拉框选择新类别\n"
            "  · 保存后生成与图片同名的 .xml（Pascal VOC 格式）\n\n"
            "分割区域（分割模式）\n"
            "  · 依次点击添加多边形顶点，双击起点或按 Enter 完成，Esc 取消\n"
            "  · 绘制中右键点击顶点可删除该点；完成后选中多边形可拖动顶点调整位置\n"
            "  · 最多 3 个类别，分别写入 PNG 的 R / G / B 通道，图上显示配置中的类别名\n"
            "  · 选中多边形后用右侧“分割类别”下拉框或工具栏按钮修改其类别\n"
            "  · 保存后同时写入 XML 和同名 _mask.png 掩码图（与原图等尺寸）\n"
            "  · 切到下一张再返回时，分割区域会从 XML 恢复，可继续调整\n\n"
            "通用\n"
            "  · 类别定义在 config.json，改完文件约 1 秒自动同步；软件内改类别也会写回文件\n"
            "  · 滚轮缩放，中键拖拽平移，F 适应窗口\n"
            "  · 标注变动后自动保存（可在“设置”菜单关闭）\n"
            "  · 右侧“类别配置 → 分割图输出设置”可开关 PNG 自动生成、设置文件名后缀\n"
            "  · “导出全部Mask”一键为所有有分割标注的图片生成 PNG，无标注的跳过\n"
            "  · Ctrl+O 打开图片，Ctrl+Shift+O 打开文件夹，Ctrl+S 手动保存\n"
            "  · ↑ / ↓ 切换上一张 / 下一张（也可用 Ctrl+Left / Ctrl+Right）",
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于",
            f"标注工具 {VERSION}\n\n"
            "目标检测框（Pascal VOC XML）+ 语义分割（RGB 三通道 PNG）标注工具。\n"
            "配置文件：config.json",
        )
