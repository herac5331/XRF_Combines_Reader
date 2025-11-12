# spx_parser.py
# Parser für .spx-Dateien (EDXRF Spektren)

import re
import base64
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import xml.etree.ElementTree as ET


# ======================
#   Hilfsfunktionen
# ======================
_NUM = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

def _repair_spx_text(text: str) -> str:
    out = []
    for ln in text.splitlines():
        if "<Channels>" in ln and "</Channels>" not in ln:
            nums = re.findall(r"-?\d+(?:\.\d+)?", ln)
            ln = "<Channels>" + ",".join(nums) + "</Channels>"
        out.append(ln)
    return "\n".join(out)

def _to_num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    if _NUM.match(s):
        try:
            return float(s)
        except Exception:
            return None
    return None

def _localname(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag

def _load_spx_root(path: str, encoding: str = "cp1252") -> ET.Element:
    for enc in (encoding, "utf-8", "latin-1", "iso-8859-1"):
        try:
            raw = Path(path).read_bytes().decode(enc, errors="replace")
            fixed = _repair_spx_text(raw)
            return ET.fromstring(fixed)
        except Exception:
            continue
    fixed = _repair_spx_text(Path(path).read_bytes().decode("utf-8", errors="ignore"))
    return ET.fromstring(fixed)

def _find_text_anywhere(root: ET.Element, tag: str) -> Optional[str]:
    for e in root.iter(tag):
        t = (e.text or "").strip()
        if t:
            return t
    tl = tag.lower()
    for e in root.iter():
        if _localname(e.tag).lower() == tl:
            t = (e.text or "").strip()
            if t:
                return t
    return None


# ======================
#   Hauptparser
# ======================
def parse_spx_file(path: str, encoding: str = "cp1252") -> Dict[str, Any]:
    root = _load_spx_root(path, encoding=encoding)

    # Name
    spec_name = None
    for ci in root.iter():
        if _localname(ci.tag) == "ClassInstance":
            if ci.attrib.get("Type") == "TRTSpectrum" and "Name" in ci.attrib:
                spec_name = ci.attrib["Name"]
                break
    spec_name = spec_name or Path(path).stem

    # Acquisition
    acquisition = {
        "real_time_ms": _to_num(_find_text_anywhere(root, "RealTime")),
        "live_time_ms": _to_num(_find_text_anywhere(root, "LifeTime")),
        "dead_time_percent": _to_num(_find_text_anywhere(root, "DeadTime")),
    }

    # Calibration
    channel_count_txt = _find_text_anywhere(root, "ChannelCount")
    calibration = {
        "channel_count": int(float(channel_count_txt)) if channel_count_txt else None,
        "offset_keV": _to_num(_find_text_anywhere(root, "CalibAbs")),
        "gain_keV_per_ch": _to_num(_find_text_anywhere(root, "CalibLin")),
    }

    # Counts
    counts: List[int] = []
    ch_text = _find_text_anywhere(root, "Channels")
    if ch_text:
        for t in ch_text.split(","):
            t = t.strip()
            if t:
                try:
                    counts.append(int(float(t)))
                except Exception:
                    pass

    return {
        "spectrum_name": spec_name,
        "file": str(Path(path).name),
        "acquisition": acquisition,
        "calibration": calibration,
        "counts": counts,
        "counts_len": len(counts),
    }


def energy_axis_keV(rec: Dict[str, Any]) -> np.ndarray:
    cal = rec.get("calibration") or {}
    offset = float(cal.get("offset_keV") or 0.0)
    gain   = float(cal.get("gain_keV_per_ch") or 0.01)
    n = int(rec.get("counts_len") or (len(rec.get("counts") or [])))
    ch = np.arange(n, dtype=float)
    return offset + gain * ch
