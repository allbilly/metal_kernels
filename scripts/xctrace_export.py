"""Shared helpers for xctrace XML table exports. Standard library only."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional, Tuple


class Table:
    """One exported xctrace table (a <node> with <schema> and <row>s)."""

    def __init__(self, path: str) -> None:
        root = ET.parse(path).getroot()
        node = root.find(".//node")
        if node is None:
            raise ValueError("no <node> in %s" % path)
        schema = node.find("schema")
        self.schema = schema.get("name", "") if schema is not None else ""
        self.columns = (
            {col.findtext("mnemonic", ""): i for i, col in enumerate(schema.findall("col"))}
            if schema is not None
            else {}
        )
        self.rows = node.findall("row")
        self._id: Dict[str, ET.Element] = {}
        for elem in node.iter():
            eid = elem.get("id")
            if eid:
                self._id[eid] = elem

    def _resolve(self, elem: Optional[ET.Element]) -> Optional[ET.Element]:
        if elem is None:
            return None
        ref = elem.get("ref")
        if ref:
            return self._id.get(ref)
        return elem

    def fmt(self, elem: Optional[ET.Element]) -> str:
        e = self._resolve(elem)
        if e is None:
            return ""
        return e.get("fmt") or (e.text or "").strip()

    def ival(self, elem: Optional[ET.Element]) -> int:
        e = self._resolve(elem)
        if e is None:
            return 0
        for text in ((e.text or "").strip(), (e.get("fmt") or "").strip()):
            if not text:
                continue
            text = text.replace(",", "")
            if re.fullmatch(r"-?[0-9]+", text):
                return int(text)
            m = re.search(r"-?[0-9]+", text)
            if m:
                return int(m.group(0))
        return 0

    def child(self, row: ET.Element, tag: str) -> Optional[ET.Element]:
        for c in row:
            if c.tag == tag:
                return c
        return None

    def children(self, row: ET.Element, tag: str) -> List[ET.Element]:
        return [c for c in row if c.tag == tag]

    def labels(self, row: ET.Element) -> List[str]:
        return [self.fmt(c) for c in self.children(row, "metal-object-label")]

    def process(self, row: ET.Element) -> str:
        return self.fmt(self.child(row, "process"))


def base_shader_name(label: str) -> str:
    """Drop the trailing specialization counter, e.g. 'steel_gemm (5)'."""
    return re.sub(r"\s+\([0-9]+\)$", "", label or "").strip()


def pick_process(
    table: Table, rows: List[ET.Element], want: Optional[str] = None
) -> Tuple[List[ET.Element], str]:
    """Filter rows to one process; default to the process with most rows."""
    names = Counter(table.process(row) for row in rows)
    names.pop("", None)
    if want:
        chosen = [name for name in names if want.lower() in name.lower()]
        if not chosen:
            available = ", ".join(sorted(names)) or "<none>"
            raise SystemExit("no process matching %r (have: %s)" % (want, available))
        name = chosen[0]
        return [row for row in rows if table.process(row) == name], name
    if not names:
        return rows, ""
    name = names.most_common(1)[0][0]
    return [row for row in rows if table.process(row) == name], name


def short(text: str, limit: int = 64) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"
