"""交互式标注画布：目标框与分割多边形的绘制、选择、移动与缩放。"""

import hashlib
from math import hypot
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from models import Annotation, Box, Polygon

BOX_COLOR = QColor("#38bdf8")
SELECT_COLOR = QColor("#fbbf24")
# 检测框按类别分配颜色，同一类别在任何图片上都保持一致
# 色相每隔 30 度一档，并交替使用深浅亮度，避免相邻类别颜色相近
BOX_PALETTE = [
    QColor("#D91212"), QColor("#E7A86A"),
    QColor("#D9D912"), QColor("#A8E76A"),
    QColor("#12D912"), QColor("#6AE7A8"),
    QColor("#12D9D9"), QColor("#6AA8E7"),
    QColor("#1212D9"), QColor("#A86AE7"),
    QColor("#D912D9"), QColor("#E76AA8"),
]
# 分割显示颜色：R / G / B 三通道
SEG_COLORS = [QColor("#ef4444"), QColor("#22c55e"), QColor("#3b82f6")]
HANDLE_SIZE = 9
Point = Tuple[float, float]


def color_for_label(label: str) -> QColor:
    """根据类别名稳定地返回一种显示颜色。"""
    digest = hashlib.md5((label or "").encode("utf-8")).digest()
    return QColor(BOX_PALETTE[digest[0] % len(BOX_PALETTE)])


def contrast_text_hex(color: QColor) -> str:
    """根据背景亮度返回黑/白文字色，保证浅色类别的标签也可读。"""
    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    return "#111827" if luminance > 150 else "#ffffff"


def _point_in_polygon(point: Point, points: List[Point]) -> bool:
    """射线法判断点是否在多边形内。"""
    x, y = point
    inside = False
    n = len(points)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class AnnotationCanvas(QGraphicsView):
    annotationChanged = pyqtSignal()
    selectionChanged = pyqtSignal(object)
    cursorMoved = pyqtSignal(float, float)
    statusHint = pyqtSignal(str)
    prevImageRequested = pyqtSignal()
    nextImageRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene_.addItem(self.image_item)

        self.setRenderHint(QPainter.Antialiasing)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.annotation = Annotation()
        self.mode = "detect"  # detect / segment
        self.box_label = "object"
        self.seg_class = 0
        self.seg_class_count = 3
        self.seg_class_names: List[str] = ["类别R", "类别G", "类别B"]

        self.selected: Optional[Tuple[str, int]] = None  # ("box"|"poly", index)
        self._drag = None  # new_box / move / resize / pan
        self._start: Optional[Point] = None
        self._draft_box: Optional[Box] = None
        self._draft_points: List[Point] = []
        self._cursor: Point = (0.0, 0.0)
        self._resize_handle: Optional[str] = None
        self._move_origin: Optional[Point] = None
        self._box_before_drag: Optional[Box] = None
        self._poly_before_drag: Optional[List[Point]] = None
        self._vertex_index: Optional[int] = None
        self._pan_last = None

    # ---------- 公共接口 ----------

    def set_seg_class_names(self, names: List[str]) -> None:
        self.seg_class_names = list(names)
        self.redraw()

    def set_annotation(self, annotation: Annotation) -> None:
        self.annotation = annotation
        self.selected = None
        self._draft_box = None
        self._draft_points = []
        self._drag = None
        if annotation.image_path and annotation.image_path.exists():
            self.image_item.setPixmap(QPixmap(str(annotation.image_path)))
        else:
            self.image_item.setPixmap(QPixmap())
        self.scene_.setSceneRect(self.image_item.boundingRect())
        self.selectionChanged.emit(None)
        self.redraw()
        self.fit_image()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._draft_box = None
        self._draft_points = []
        self._drag = None
        self.statusHint.emit(
            "检测模式：拖拽鼠标画框，点击可选中并移动，四角可缩放"
            if mode == "detect"
            else "分割模式：依次点击添加顶点，双击/Enter 完成，Esc 取消"
        )
        self.redraw()

    def fit_image(self) -> None:
        if self.image_item.pixmap().isNull():
            return
        self.fitInView(self.image_item.boundingRect(), Qt.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def select(self, selection: Optional[Tuple[str, int]]) -> None:
        if selection != self.selected:
            self.selected = selection
            self.selectionChanged.emit(selection)
            self.redraw()

    def delete_selected(self) -> None:
        if not self.selected:
            return
        kind, index = self.selected
        if kind == "box" and index < len(self.annotation.boxes):
            del self.annotation.boxes[index]
        elif kind == "poly" and index < len(self.annotation.polygons):
            del self.annotation.polygons[index]
        self.selected = None
        self.annotationChanged.emit()
        self.selectionChanged.emit(None)
        self.redraw()

    def clear_annotations(self) -> None:
        self.annotation.clear()
        self.selected = None
        self._draft_points = []
        self._draft_box = None
        self.annotationChanged.emit()
        self.selectionChanged.emit(None)
        self.redraw()

    def finish_current_polygon(self) -> None:
        if len(self._draft_points) >= 3:
            poly = Polygon(list(self._draft_points), self.seg_class)
            self.annotation.polygons.append(poly)
            self.select(("poly", len(self.annotation.polygons) - 1))
            self.annotationChanged.emit()
            name = (
                self.seg_class_names[self.seg_class]
                if self.seg_class < len(self.seg_class_names)
                else f"类别 {self.seg_class + 1}"
            )
            self.statusHint.emit(f"已添加分割区域：{name}")
        self._draft_points = []
        self.redraw()

    def _remove_vertex_at(self, point: Point) -> None:
        """右键删除正在绘制的点，或任意已标注多边形的顶点（至少保留 3 个点）。"""
        radius = self._pixel_radius(10)
        if self._draft_points:
            index = self._nearest_vertex(self._draft_points, point, radius)
            if index is not None:
                self._draft_points.pop(index)
                self.statusHint.emit(f"已删除顶点，剩余 {len(self._draft_points)} 个点")
                self.redraw()
                return

        for i in range(len(self.annotation.polygons) - 1, -1, -1):
            points = self.annotation.polygons[i].points
            if len(points) <= 3:
                continue
            index = self._nearest_vertex(points, point, radius)
            if index is not None:
                del points[index]
                self.select(("poly", i))
                self.annotationChanged.emit()
                self.statusHint.emit(f"已删除顶点，剩余 {len(points)} 个点")
                self.redraw()
                return

    # ---------- 坐标与命中测试 ----------

    def _scene_point(self, screen_pos) -> Point:
        p = self.mapToScene(screen_pos)
        return p.x(), p.y()

    def _clamp(self, point: Point) -> Point:
        w, h = self.annotation.width, self.annotation.height
        return (
            min(max(point[0], 0.0), float(w)),
            min(max(point[1], 0.0), float(h)),
        )

    def _pixel_radius(self, size: float) -> float:
        scale = self.transform().m11() or 1.0
        return size / scale

    def _hit_box(self, point: Point) -> Optional[int]:
        for i in range(len(self.annotation.boxes) - 1, -1, -1):
            b = self.annotation.boxes[i].ordered()
            if b.xmin <= point[0] <= b.xmax and b.ymin <= point[1] <= b.ymax:
                return i
        return None

    def _hit_polygon(self, point: Point) -> Optional[int]:
        for i in range(len(self.annotation.polygons) - 1, -1, -1):
            if _point_in_polygon(point, self.annotation.polygons[i].points):
                return i
        return None

    def _hit_handle(self, point: Point) -> Optional[str]:
        if not self.selected or self.selected[0] != "box":
            return None
        if self.selected[1] >= len(self.annotation.boxes):
            return None
        b = self.annotation.boxes[self.selected[1]].ordered()
        handles = {
            "tl": (b.xmin, b.ymin),
            "tr": (b.xmax, b.ymin),
            "bl": (b.xmin, b.ymax),
            "br": (b.xmax, b.ymax),
        }
        radius = self._pixel_radius(HANDLE_SIZE)
        for name, (hx, hy) in handles.items():
            if abs(point[0] - hx) <= radius and abs(point[1] - hy) <= radius:
                return name
        return None

    def _nearest_vertex(self, points: List[Point], point: Point, radius: float) -> Optional[int]:
        """返回 radius 范围内距离 point 最近的顶点索引。"""
        best_index, best_distance = None, radius
        for i, (x, y) in enumerate(points):
            distance = hypot(point[0] - x, point[1] - y)
            if distance <= best_distance:
                best_index, best_distance = i, distance
        return best_index

    def _hit_polygon_vertex(self, point: Point) -> Optional[Tuple[int, int]]:
        """返回 (多边形索引, 顶点索引)，无需先选中多边形。"""
        radius = self._pixel_radius(HANDLE_SIZE)
        for i in range(len(self.annotation.polygons) - 1, -1, -1):
            index = self._nearest_vertex(self.annotation.polygons[i].points, point, radius)
            if index is not None:
                return i, index
        return None

    # ---------- 鼠标事件 ----------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._drag = "pan"
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.RightButton:
            if self.mode == "segment":
                self._remove_vertex_at(self._clamp(self._scene_point(event.pos())))
            return
        if event.button() != Qt.LeftButton:
            return

        point = self._clamp(self._scene_point(event.pos()))
        self._cursor = point

        if self.mode == "detect":
            handle = self._hit_handle(point)
            if handle:
                self._drag = "resize"
                self._resize_handle = handle
                self._box_before_drag = self.annotation.boxes[self.selected[1]].ordered()
                return

            index = self._hit_box(point)
            if index is not None:
                self.select(("box", index))
                self._drag = "move"
                self._move_origin = point
                self._box_before_drag = Box(**vars(self.annotation.boxes[index]))
                return

            self.select(None)
            self._start = point
            self._draft_box = Box(point[0], point[1], point[0], point[1], self.box_label)
            self._drag = "new_box"

        else:  # segment
            vertex = self._hit_polygon_vertex(point)
            if vertex is not None:
                self.select(("poly", vertex[0]))
                self._drag = "move_vertex"
                self._vertex_index = vertex[1]
                return

            if not self._draft_points:
                index = self._hit_polygon(point)
                if index is not None:
                    self.select(("poly", index))
                    self._drag = "move"
                    self._move_origin = point
                    self._poly_before_drag = self.annotation.polygons[index].copy_points()
                    return

            # 避免双击产生的重复顶点
            if self._draft_points and hypot(
                point[0] - self._draft_points[-1][0],
                point[1] - self._draft_points[-1][1],
            ) < self._pixel_radius(4):
                return

            if self._draft_points and len(self._draft_points) >= 3:
                first = self._draft_points[0]
                if hypot(point[0] - first[0], point[1] - first[1]) < self._pixel_radius(10):
                    self.finish_current_polygon()
                    return

            self.select(None)
            self._draft_points.append(point)
            self.statusHint.emit(
                f"已添加 {len(self._draft_points)} 个顶点；双击或 Enter 完成，Esc 取消"
            )
        self.redraw()

    def mouseMoveEvent(self, event) -> None:
        raw = self._scene_point(event.pos())
        self._cursor = raw
        self.cursorMoved.emit(raw[0], raw[1])

        if self._drag == "pan":
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return

        point = self._clamp(raw)
        if self._drag == "new_box" and self._draft_box is not None:
            self._draft_box.xmax, self._draft_box.ymax = point
        elif self._drag == "move":
            self._move_selected(point)
        elif self._drag == "resize":
            self._resize_selected(point)
        elif self._drag == "move_vertex":
            self._move_vertex(point)
        self.redraw()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag == "pan":
            self._drag = None
            self.unsetCursor()
            return
        if event.button() != Qt.LeftButton:
            return

        if self._drag == "new_box" and self._draft_box is not None:
            box = self._draft_box.ordered()
            if box.xmax - box.xmin >= 3 and box.ymax - box.ymin >= 3:
                self.annotation.boxes.append(box)
                self.select(("box", len(self.annotation.boxes) - 1))
                self.annotationChanged.emit()
                self.statusHint.emit(f"已添加目标框：{box.label}")
            self._draft_box = None
        elif self._drag in ("move", "resize", "move_vertex"):
            self.annotationChanged.emit()

        self._drag = None
        self._vertex_index = None
        self._start = None
        self._box_before_drag = None
        self._poly_before_drag = None
        self._resize_handle = None
        self.redraw()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.mode == "segment" and len(self._draft_points) >= 3:
            self.finish_current_polygon()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        old_pos = self.mapToScene(event.pos())
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._draft_points = []
            self._draft_box = None
            self._drag = None
            self.select(None)
            self.statusHint.emit("已取消当前绘制")
            self.redraw()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.mode == "segment":
                self.finish_current_polygon()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif event.key() == Qt.Key_Up:
            self.prevImageRequested.emit()
        elif event.key() == Qt.Key_Down:
            self.nextImageRequested.emit()
        else:
            super().keyPressEvent(event)

    # ---------- 编辑操作 ----------

    def _move_selected(self, point: Point) -> None:
        if not self.selected or not self._move_origin:
            return
        dx = point[0] - self._move_origin[0]
        dy = point[1] - self._move_origin[1]
        w, h = self.annotation.width, self.annotation.height

        if self.selected[0] == "box" and self._box_before_drag is not None:
            b = self._box_before_drag
            dx = min(max(dx, -min(b.xmin, b.xmax)), w - max(b.xmin, b.xmax))
            dy = min(max(dy, -min(b.ymin, b.ymax)), h - max(b.ymin, b.ymax))
            box = self.annotation.boxes[self.selected[1]]
            box.xmin, box.ymin = b.xmin + dx, b.ymin + dy
            box.xmax, box.ymax = b.xmax + dx, b.ymax + dy
        elif self.selected[0] == "poly" and self._poly_before_drag:
            xs = [p[0] for p in self._poly_before_drag]
            ys = [p[1] for p in self._poly_before_drag]
            dx = min(max(dx, -min(xs)), w - max(xs))
            dy = min(max(dy, -min(ys)), h - max(ys))
            poly = self.annotation.polygons[self.selected[1]]
            poly.points = [(x + dx, y + dy) for x, y in self._poly_before_drag]

    def _resize_selected(self, point: Point) -> None:
        if (
            not self.selected
            or self.selected[0] != "box"
            or self._box_before_drag is None
            or self._resize_handle is None
        ):
            return
        box = self.annotation.boxes[self.selected[1]]
        if "l" in self._resize_handle:
            box.xmin = point[0]
        if "r" in self._resize_handle:
            box.xmax = point[0]
        if "t" in self._resize_handle:
            box.ymin = point[1]
        if "b" in self._resize_handle:
            box.ymax = point[1]

    def _move_vertex(self, point: Point) -> None:
        if (
            not self.selected
            or self.selected[0] != "poly"
            or self._vertex_index is None
            or self.selected[1] >= len(self.annotation.polygons)
        ):
            return
        points = self.annotation.polygons[self.selected[1]].points
        if self._vertex_index < len(points):
            points[self._vertex_index] = point

    # ---------- 绘制 ----------

    def _cosmetic_pen(self, color: QColor, width: float, style=Qt.SolidLine) -> QPen:
        pen = QPen(color, width, style)
        pen.setCosmetic(True)
        return pen

    def redraw(self) -> None:
        for item in self.scene_.items():
            if item is not self.image_item:
                self.scene_.removeItem(item)

        if self.image_item.pixmap().isNull():
            return

        self._draw_boxes()
        self._draw_polygons()
        self._draw_draft()

    def _draw_boxes(self) -> None:
        for i, box in enumerate(self.annotation.boxes):
            b = box.ordered()
            selected = self.selected == ("box", i)
            color = color_for_label(box.label)
            if selected:
                offset = self._pixel_radius(4)
                outline = QGraphicsRectItem(
                    QRectF(
                        b.xmin - offset,
                        b.ymin - offset,
                        b.xmax - b.xmin + offset * 2,
                        b.ymax - b.ymin + offset * 2,
                    )
                )
                outline.setPen(self._cosmetic_pen(QColor("#ffffff"), 2, Qt.DashLine))
                self.scene_.addItem(outline)
            rect = QGraphicsRectItem(
                QRectF(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin)
            )
            rect.setPen(self._cosmetic_pen(color, 3 if selected else 2))
            fill = QColor(color)
            fill.setAlpha(38 if selected else 26)
            rect.setBrush(QBrush(fill))
            self.scene_.addItem(rect)

            label = QGraphicsTextItem(box.label)
            label.setDefaultTextColor(QColor(contrast_text_hex(color)))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            label.setFont(font)
            label_y = b.ymin - 20 if b.ymin >= 20 else b.ymin
            label.setPos(b.xmin, label_y)
            width = label.boundingRect().width() + 8
            bg = QGraphicsRectItem(QRectF(b.xmin, label_y, width, 18))
            bg.setBrush(QBrush(color))
            bg.setPen(QPen(Qt.NoPen))
            self.scene_.addItem(bg)
            self.scene_.addItem(label)

            if selected:
                radius = self._pixel_radius(HANDLE_SIZE)
                for hx, hy in (
                    (b.xmin, b.ymin),
                    (b.xmax, b.ymin),
                    (b.xmin, b.ymax),
                    (b.xmax, b.ymax),
                ):
                    handle = QGraphicsEllipseItem(
                        QRectF(hx - radius / 2, hy - radius / 2, radius, radius)
                    )
                    handle.setBrush(QBrush(QColor("#ffffff")))
                    handle.setPen(self._cosmetic_pen(color, 2))
                    self.scene_.addItem(handle)

    def _draw_polygons(self) -> None:
        for i, poly in enumerate(self.annotation.polygons):
            if len(poly.points) < 3:
                continue
            selected = self.selected == ("poly", i)
            color = SEG_COLORS[poly.class_id] if poly.class_id < len(SEG_COLORS) else SEG_COLORS[0]
            item = QGraphicsPolygonItem(
                QPolygonF([QPointF(x, y) for x, y in poly.points])
            )
            item.setPen(self._cosmetic_pen(color, 3 if selected else 2))
            fill = QColor(color)
            fill.setAlpha(70 if selected else 40)
            item.setBrush(QBrush(fill))
            self.scene_.addItem(item)

            cx = sum(p[0] for p in poly.points) / len(poly.points)
            cy = sum(p[1] for p in poly.points) / len(poly.points)
            label = (
                self.seg_class_names[poly.class_id]
                if poly.class_id < len(self.seg_class_names)
                else f"类别{poly.class_id + 1}"
            )
            text = QGraphicsTextItem(label)
            text.setDefaultTextColor(QColor("#ffffff"))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            text.setFont(font)
            tx = cx - text.boundingRect().width() / 2
            ty = cy - 9
            text.setPos(tx, ty)
            background = QGraphicsRectItem(
                QRectF(tx - 4, ty - 2, text.boundingRect().width() + 8, 20)
            )
            background.setBrush(QBrush(color))
            background.setPen(QPen(Qt.NoPen))
            self.scene_.addItem(background)
            self.scene_.addItem(text)

            if selected:
                radius = self._pixel_radius(HANDLE_SIZE)
                for px, py in poly.points:
                    handle = QGraphicsEllipseItem(
                        QRectF(px - radius / 2, py - radius / 2, radius, radius)
                    )
                    handle.setBrush(QBrush(QColor("#ffffff")))
                    handle.setPen(self._cosmetic_pen(color, 2))
                    self.scene_.addItem(handle)

    def _draw_draft(self) -> None:
        if self._draft_box is not None:
            b = self._draft_box.ordered()
            rect = QGraphicsRectItem(
                QRectF(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin)
            )
            rect.setPen(
                self._cosmetic_pen(color_for_label(self._draft_box.label), 2, Qt.DashLine)
            )
            self.scene_.addItem(rect)

        if not self._draft_points:
            return

        color = SEG_COLORS[self.seg_class] if self.seg_class < len(SEG_COLORS) else SEG_COLORS[0]
        for start, end in zip(self._draft_points, self._draft_points[1:]):
            line = QGraphicsLineItem(QLineF(start[0], start[1], end[0], end[1]))
            line.setPen(self._cosmetic_pen(color, 2))
            self.scene_.addItem(line)

        if len(self._draft_points) >= 2:
            last = self._draft_points[-1]
            line = QGraphicsLineItem(
                QLineF(last[0], last[1], self._cursor[0], self._cursor[1])
            )
            line.setPen(self._cosmetic_pen(color, 1, Qt.DashLine))
            self.scene_.addItem(line)

        radius = self._pixel_radius(5)
        for px, py in self._draft_points:
            dot = QGraphicsEllipseItem(QRectF(px - radius, py - radius, radius * 2, radius * 2))
            dot.setBrush(QBrush(color))
            dot.setPen(self._cosmetic_pen(QColor("#ffffff"), 1))
            self.scene_.addItem(dot)
