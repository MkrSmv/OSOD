"""Разбор Pascal VOC XML — общий для HR-InsDet и PKU-Market-PCB."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def read_voc(path: Path) -> dict:
    """`{"filename", "wh", "objects": [{"name", "box"}]}`; рамка — `bndbox` без сдвига."""
    t = ET.parse(path).getroot()
    s = t.find("size")
    objs = []
    for o in t.findall("object"):
        bb = o.find("bndbox")
        objs.append({"name": o.findtext("name"),
                     "box": [int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]})
    return {"filename": t.findtext("filename"), "wh": [int(s.findtext("width")), int(s.findtext("height"))],
            "objects": objs}
