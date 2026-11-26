"""标注数据模型与全局配置。"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

VERSION = "v1.0"


@dataclass
class Box:
    """图像像素坐标系下的目标检测框。"""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "object"
    difficult: bool = False

    def ordered(self) -> "Box":
        """返回左上/右下顺序规范化后的副本。"""
        return Box(
            min(self.xmin, self.xmax),
            min(self.ymin, self.ymax),
            max(self.xmin, self.xmax),
            max(self.ymin, self.ymax),
            self.label,
            self.difficult,
        )

    def area(self) -> float:
        b = self.ordered()
        return (b.xmax - b.xmin) * (b.ymax - b.ymin)


@dataclass
class Polygon:
    """分割多边形，class_id 对应 RGB 中的通道：0=R, 1=G, 2=B。"""

    points: List[Tuple[float, float]] = field(default_factory=list)
    class_id: int = 0

    def copy_points(self) -> List[Tuple[float, float]]:
        return list(self.points)


@dataclass
class Annotation:
    """单张图片的全部标注。"""

    image_path: Optional[Path] = None
    width: int = 0
    height: int = 0
    boxes: List[Box] = field(default_factory=list)
    polygons: List[Polygon] = field(default_factory=list)

    @property
    def image_name(self) -> str:
        return self.image_path.name if self.image_path else ""

    def clear(self) -> None:
        self.boxes.clear()
        self.polygons.clear()

    def is_empty(self) -> bool:
        return not self.boxes and not self.polygons


class Config:
    """类别与输出配置，持久化到 config.json，支持外部修改后热同步。"""

    MAX_SEG_CLASSES = 3
    DEFAULT_BOX_CLASSES = ["excavator", "people", "person", "car", "bicycle", "object"]
    DEFAULT_SEG_CLASSES = ["类别R", "类别G", "类别B"]
    CONFIG_FILE = Path(__file__).with_name("config.json")

    def __init__(self, path=None) -> None:
        self.path = Path(path) if path else self.CONFIG_FILE
        self.box_classes: List[str] = list(self.DEFAULT_BOX_CLASSES)
        self.seg_classes: List[str] = list(self.DEFAULT_SEG_CLASSES)
        self.output_dir: Optional[Path] = None
        self.auto_generate_mask: bool = True
        self.mask_suffix: str = "_mask"
        self._mtime: Optional[int] = None
        self.last_error: str = ""
        self.load(create_if_missing=True)

    def load(self, create_if_missing: bool = False) -> bool:
        """从配置文件重新加载类别，返回是否成功。失败时保留当前设置且不写回文件。"""
        self.last_error = ""
        if not self.path.exists():
            if create_if_missing:
                self.save()
                return True
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.last_error = f"配置文件读取失败: {exc}"
            self._mtime = self.path.stat().st_mtime_ns
            return False
        if not isinstance(data, dict):
            self.last_error = "配置文件根节点必须是 JSON 对象"
            self._mtime = self.path.stat().st_mtime_ns
            return False

        boxes = data.get("box_classes")
        if isinstance(boxes, list):
            names = [str(name).strip() for name in boxes if str(name).strip()]
            if names:
                self.box_classes = names

        segs = data.get("seg_classes")
        if isinstance(segs, list):
            names = [str(name).strip() for name in segs[: self.MAX_SEG_CLASSES]]
            names = [name or default for name, default in zip(names, self.DEFAULT_SEG_CLASSES)]
            while len(names) < self.MAX_SEG_CLASSES:
                names.append(self.DEFAULT_SEG_CLASSES[len(names)])
            self.seg_classes = names[: self.MAX_SEG_CLASSES]

        output_dir = data.get("output_dir")
        self.output_dir = Path(output_dir) if output_dir else None

        auto_mask = data.get("auto_generate_mask")
        if isinstance(auto_mask, bool):
            self.auto_generate_mask = auto_mask

        suffix = data.get("mask_suffix")
        if isinstance(suffix, str):
            self.mask_suffix = suffix.strip()

        self._mtime = self.path.stat().st_mtime_ns
        return True

    def save(self) -> None:
        """把当前配置写回配置文件。"""
        data = {
            "box_classes": self.box_classes,
            "seg_classes": self.seg_classes,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "auto_generate_mask": self.auto_generate_mask,
            "mask_suffix": self.mask_suffix,
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self._mtime = self.path.stat().st_mtime_ns

    def changed_externally(self) -> bool:
        """检测配置文件是否被外部修改。"""
        if not self.path.exists():
            return False
        return self.path.stat().st_mtime_ns != self._mtime
