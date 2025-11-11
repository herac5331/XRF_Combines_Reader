import xml.etree.ElementTree as ET
from typing import Dict, Any
from constants import ELEMENTS, LINE_MAP, ATMOSPHERE_MAP

def parse_xadf_safe(root: ET.Element) -> Dict[str, Any]:
    """
    Fault-toleranter Parser für Bruker XMethod XADF Dateien.
    Gibt ein Dictionary mit Info, MeasurementParameters, Layers etc. zurück.
    """
    parsed = {
        "Info": {},
        "MeasurementParameters": {},
        "CalculationParameters": {},
        "Elements": [],
        "Layers": [],
        "_errors": []
    }

def translate_used_lines(s):
    if not s: return []
    try: return [LINE_MAP.get(int(x),f"Line{x}") for x in s.split(",") if x.strip()]
    except: return []

def element_to_dict(elem):
    d={}
    for c in elem:
        if c.tag=="ClassInstance":
            for sub in c: d[sub.tag]=element_to_dict(sub)
            continue
        d[c.tag]=element_to_dict(c) if len(c) else (c.text.strip() if c.text else None)
    return d

# ---------- Safe Parser ----------
def parse_xadf_safe(root):
    """Completely fault-tolerant XADF parser with extended metadata extraction."""
    parsed = {
        "Info": {},
        "MeasurementParameters": {},
        "CalculationParameters": {},
        "Elements": [],
        "Layers": [],
        "_errors": []
    }

    def safe_get_dict(elem):
        try:
            return element_to_dict(elem) if elem is not None else {}
        except Exception as e:
            parsed["_errors"].append(str(e))
            return {}

    # --- Info section ---
    try:
        info_block = root.find(".//ClassInstance[@Type='TXS2_XADFMgr_Info']")
        if info_block is not None:
            info_dict = safe_get_dict(info_block)
            parsed["Info"] = info_dict

            # Extract key fields from Info
            info_data = info_dict.get("Info", info_dict)
            parsed["InfoExtracted"] = {
                "SpectrumProcessingType": info_data.get("SpectrumProcessingType"),
                "AnalysisMethod": info_data.get("AnalysisMethod"),
                "ModifyDate": info_data.get("ModifyDate"),
                "ModifyDateSerialData": info_data.get("ModifyDateSerialData"),
                "CalibDate": info_data.get("CalibDate"),
                "CalibDateSerialData": info_data.get("CalibDateSerialData")
            }
    except Exception as e:
        parsed["_errors"].append(f"Info section: {e}")

    # --- Measurement parameters ---
    try:
        m = root.find(".//ClassInstance[@Type='TXS2_XADFMgr_MParam']")
        if m is not None:
            mdata = safe_get_dict(m)
            mp = mdata.get("MParam", mdata)
            z = mp.get("TubeZ")
            if z and z.isdigit():
                mp["TubeElement"] = ELEMENTS.get(int(z), "?")
            atm = mp.get("Atmosphere")
            if atm in ATMOSPHERE_MAP:
                mp["AtmosphereName"] = ATMOSPHERE_MAP[atm]
            parsed["MeasurementParameters"] = mp

            # Extract UnitType and DetectorType
            parsed["MeasurementMeta"] = {
                "UnitType": mp.get("UnitType"),
                "DetectorType": mp.get("DetectorType")
            }
    except Exception as e:
        parsed["_errors"].append(f"Measurement parameters: {e}")

    # --- Calculation parameters ---
    try:
        calc = root.find(".//ClassInstance[@Type='TXS2_XADFMgr_CalcParam']")
        if calc is not None:
            cdata = safe_get_dict(calc)
            parsed["CalculationParameters"] = cdata.get("CalculationParameters", cdata)
    except Exception as e:
        parsed["_errors"].append(f"Calculation parameters: {e}")

    # --- Elements (same as before) ---
    try:
        elements = []
        for e in root.findall(".//ClassInstance[@Type='TXS2_XADFMgr_SingleElement']"):
            ed = safe_get_dict(e)
            if not ed: continue
            se = next(iter(ed.values())) if len(ed)==1 and list(ed.keys())[0].startswith("SingleElement_") else ed
            if not isinstance(se,dict): continue
            z = se.get("Z")
            if z and z.isdigit(): se["ElementSymbol"]=ELEMENTS.get(int(z),"?")
            pe_num=None
            for v in se.values():
                if isinstance(v,dict) and "PE_Spc_Number" in v:
                    pe_num=v["PE_Spc_Number"]; break
            se["PE_Spc_Number"]=pe_num or "?"
            for v in se.values():
                if isinstance(v,dict) and "UsedLines" in v:
                    v["EmissionLines"]=translate_used_lines(v["UsedLines"])
            elements.append(se)
        parsed["Elements"]=elements
    except Exception as e:
        parsed["_errors"].append(f"Elements: {e}")

    # --- Layers (same as before) ---
    try:
        layers=[]
        for i,lyr in enumerate(root.findall(".//ClassInstance[@Type='TXS2_XADFMgr_SingleLayer']"),1):
            L=safe_get_dict(lyr)
            if not L: continue
            lname=next(iter(L.values())) if isinstance(L,dict) and len(L) else L
            if not isinstance(lname,dict): continue
            layer_info={"Index":i,
                        "Description":lname.get("Description"),
                        "Thickness_um":lname.get("Thickness"),
                        "Density_gcm3":lname.get("Density",{}).get("Default")
                            if isinstance(lname.get("Density"),dict) else lname.get("Density"),
                        "Elements":[]}
            for key,val in lname.items():
                if key.startswith("Element_") and isinstance(val,dict):
                    gi=val.get("GlobalElementIndex")
                    ei=None
                    try: ei=parsed["Elements"][int(gi)]
                    except Exception: ei={}
                    sym=ei.get("ElementSymbol","?")
                    conc=val.get("StartConcentration","?")
                    pe=ei.get("PE_Spc_Number","?")
                    lines=[]
                    for sub in ei.values():
                        if isinstance(sub,dict) and "EmissionLines" in sub:
                            lines=sub["EmissionLines"]
                    layer_info["Elements"].append({
                        "Symbol":sym,
                        "Conc":conc,
                        "PE_Spc_Number":pe,
                        "Lines":lines
                    })
            layers.append(layer_info)
        parsed["Layers"]=layers
    except Exception as e:
        parsed["_errors"].append(f"Layers: {e}")

    return parsed
