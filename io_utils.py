"""图片加载与标注文件导出：Pascal VOC XML 与 RGB 通道 PNG 掩码。"""

from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from models import Annotation, Box, Polygon

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# OpenCV 使用 BGR 顺序：class 0 -> 红通道，1 -> 绿通道，2 -> 蓝通道
MASK_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
]


def list_images(folder, recursive: bool = True) -> List[Path]:
    """列出文件夹中的图片，默认递归扫描所有子目录。"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: str(p.relative_to(folder)).lower(),
    )


def read_boxes_from_xml(xml_path) -> List[Box]:
    """读取已有的 Pascal VOC XML 中目标框；文件不存在或格式错误时返回空列表。"""
    path = Path(xml_path)
    if not path.exists():
        return []
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError):
        return []

    boxes: List[Box] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        def number(tag: str) -> float:
            element = bndbox.find(tag)
            if element is None or element.text is None:
                return 0.0
            try:
                return float(element.text)
            except ValueError:
                return 0.0

        xmin, ymin = number("xmin"), number("ymin")
        xmax, ymax = number("xmax"), number("ymax")
        if xmax <= xmin or ymax <= ymin:
            continue

        name_element = obj.find("name")
        label = name_element.text.strip() if name_element is not None and name_element.text else "object"
        difficult_element = obj.find("difficult")
        difficult = difficult_element is not None and difficult_element.text == "1"
        boxes.append(Box(xmin, ymin, xmax, ymax, label, difficult))
    return boxes


def read_polygons_from_xml(xml_path) -> List[Polygon]:
    """读取 XML 中 <segmentation> 里的多边形顶点。"""
    path = Path(xml_path)
    if not path.exists():
        return []
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError):
        return []

    section = root.find("segmentation")
    if section is None:
        return []
    polygons: List[Polygon] = []
    for region in section.findall("region"):
        try:
            class_id = int(region.get("class_id", "0"))
        except (TypeError, ValueError):
            class_id = 0
        points = []
        for point in region.findall("point"):
            try:
                points.append((float(point.get("x")), float(point.get("y"))))
            except (TypeError, ValueError):
                continue
        if len(points) >= 3:
            polygons.append(Polygon(points, class_id))
    return polygons


def read_polygons_from_mask(mask_path) -> List[Polygon]:
    """兼容旧版：XML 中没有分割信息时，从 RGB 通道掩码轮廓恢复多边形。"""
    path = Path(mask_path)
    if not path.exists():
        return []
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return []

    polygons: List[Polygon] = []
    for class_id, color in enumerate(MASK_COLORS_BGR):
        channel_mask = np.all(image == np.array(color, dtype=np.uint8), axis=2)
        contours, _ = cv2.findContours(
            channel_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if cv2.contourArea(contour) < 4:
                continue
            approx = cv2.approxPolyDP(contour, 2.0, True)
            points = [(float(p[0][0]), float(p[0][1])) for p in approx]
            if len(points) >= 3:
                polygons.append(Polygon(points, class_id))
    return polygons


def load_polygons_for_image(image_path, output_dir=None, mask_suffix: str = "_mask") -> List[Polygon]:
    """读取图片的分割标注：优先 XML，其次按配置的掩码文件，最后兼容旧版 _mask。"""
    path = Path(image_path)
    polygons = read_polygons_from_xml(path.with_suffix(".xml"))
    if polygons:
        return polygons
    configured = mask_path_for(path, output_dir, mask_suffix)
    legacy = path.with_name(f"{path.stem}_mask.png")
    polygons = read_polygons_from_mask(configured)
    if not polygons and configured != legacy:
        polygons = read_polygons_from_mask(legacy)
    return polygons


def read_image_size(path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def write_voc_xml(
    annotation: Annotation,
    xml_path,
    folder_name: str = "",
    seg_labels: Optional[List[str]] = None,
) -> None:
    """按 Pascal VOC 格式写出目标框与分割多边形标注。"""
    root = ET.Element("annotation")
    image_path = annotation.image_path

    def add(parent, tag: str, value) -> ET.Element:
        child = ET.SubElement(parent, tag)
        child.text = str(value)
        return child

    add(root, "folder", folder_name or (image_path.parent.name if image_path else "images"))
    add(root, "filename", image_path.name if image_path else "image")
    add(root, "path", str(image_path) if image_path else "")

    source = ET.SubElement(root, "source")
    add(source, "database", "Unknown")

    size = ET.SubElement(root, "size")
    add(size, "width", int(annotation.width))
    add(size, "height", int(annotation.height))
    add(size, "depth", 3)

    add(root, "segmented", 1 if annotation.polygons else 0)

    for box in annotation.boxes:
        obj = ET.SubElement(root, "object")
        add(obj, "name", box.label)
        add(obj, "pose", "Unspecified")
        add(obj, "truncated", 0)
        add(obj, "difficult", 1 if box.difficult else 0)
        bndbox = ET.SubElement(obj, "bndbox")
        b = box.ordered()
        add(bndbox, "xmin", int(round(b.xmin)))
        add(bndbox, "ymin", int(round(b.ymin)))
        add(bndbox, "xmax", int(round(b.xmax)))
        add(bndbox, "ymax", int(round(b.ymax)))

    if annotation.polygons:
        segmentation = ET.SubElement(root, "segmentation")
        for poly in annotation.polygons:
            label = (
                seg_labels[poly.class_id]
                if seg_labels and poly.class_id < len(seg_labels)
                else f"类别{poly.class_id + 1}"
            )
            region = ET.SubElement(
                segmentation, "region", {"class_id": str(poly.class_id), "label": label}
            )
            for x, y in poly.points:
                ET.SubElement(
                    region, "point", {"x": str(int(round(x))), "y": str(int(round(y)))}
                )

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(str(xml_path), encoding="utf-8", xml_declaration=True)


def write_mask_png(annotation: Annotation, png_path) -> None:
    """写出与原图等尺寸的 PNG 掩码，3 个类别分别写入 R/G/B 通道。"""
    h, w = int(annotation.height), int(annotation.width)
    if h <= 0 or w <= 0:
        raise ValueError("图片尺寸无效，无法导出掩码")

    mask = np.zeros((h, w, 3), dtype=np.uint8)  # BGR
    for poly in annotation.polygons:
        if len(poly.points) < 3:
            continue
        pts = np.array(
            [
                [min(max(int(round(x)), 0), w - 1), min(max(int(round(y)), 0), h - 1)]
                for x, y in poly.points
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        color = (
            MASK_COLORS_BGR[poly.class_id]
            if 0 <= poly.class_id < len(MASK_COLORS_BGR)
            else MASK_COLORS_BGR[0]
        )
        cv2.fillPoly(mask, [pts], color)

    rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(png_path))


def mask_path_for(image_path, output_dir=None, suffix: str = "_mask") -> Path:
    """根据图片名和后缀生成掩码路径；若与源图片相同则追加 _mask 避免覆盖原图。"""
    source = Path(image_path)
    out = Path(output_dir) if output_dir else source.parent
    suffix = (suffix or "").strip()
    candidate = out / f"{source.stem}{suffix}.png"
    if candidate.resolve() == source.resolve():
        candidate = out / f"{source.stem}_mask.png"
    return candidate


def save_annotation(
    annotation: Annotation,
    output_dir: Optional[Path] = None,
    force_xml: bool = False,
    force_mask: bool = False,
    seg_labels: Optional[List[str]] = None,
    auto_generate_mask: bool = True,
    mask_suffix: str = "_mask",
) -> Tuple[Optional[Path], Optional[Path]]:
    """保存标注，返回 (xml_path, mask_path)，未生成的文件为 None。"""
    if not annotation.image_path:
        raise ValueError("尚未打开图片")

    out = Path(output_dir) if output_dir else annotation.image_path.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = annotation.image_path.stem
    xml_path = out / f"{stem}.xml"
    mask_path = mask_path_for(annotation.image_path, output_dir, mask_suffix)

    xml_written = None
    mask_written = None
    if annotation.boxes or annotation.polygons or force_xml:
        write_voc_xml(annotation, xml_path, seg_labels=seg_labels)
        xml_written = xml_path
    if force_mask or (annotation.polygons and auto_generate_mask):
        write_mask_png(annotation, mask_path)
        mask_written = mask_path
    return xml_written, mask_written
