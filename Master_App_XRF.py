# master_app.py

import sys
import os

# --- Module-Pfad hinzufügen ---
# MODULE_PATH = r"D:\Profile\a5574\Python\mongoDB\XRF_combined"
# if MODULE_PATH not in sys.path:
#     sys.path.append(MODULE_PATH)

# --- Standard Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import xml.etree.ElementTree as ET
import json
import plotly.express as px

# --- Eigene Module ---
from xrf_viewer import XRFViewer
from spx_parser import parse_spx_file, energy_axis_keV
from xadf_parser import parse_xadf_safe
from constants import ELEMENTS, LINE_MAP, ATMOSPHERE_MAP, MOLAR_MASSES

# ---------------------------
# --- Master App Start ------
# ---------------------------
def main():
    st.set_page_config(page_title="XRF Analysis Suite", layout="wide")
    st.sidebar.title("🧭 Navigation")

    app_mode = st.sidebar.radio(
        "Wähle den Modus:",
        ["XRF Viewer", "SPX Spectrum Viewer", "Bruker XMethod XADF Viewer"]
    )

    # ------------------ XRF Viewer ------------------
    if app_mode == "XRF Viewer":
        viewer = XRFViewer()
        viewer.run_streamlit_app()

    # ------------------ SPX Spectrum Viewer ------------------
    elif app_mode == "SPX Spectrum Viewer":
        st.title("📈 SPX Spectrum Viewer")
    
        uploaded = st.file_uploader("Upload a .spx file", type=["spx"])
    
        if uploaded:
            import tempfile, os
    
            try:
                # Temporäre Datei speichern
                with tempfile.NamedTemporaryFile(delete=False, suffix=".spx") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
    
                # SPX-Datei parsen (jetzt mit Pfad!)
                rec = parse_spx_file(tmp_path)
    
                # Energieachse und Counts extrahieren
                energy = energy_axis_keV(rec)
                counts = rec.get("counts", [])
    
                if counts:
                    df_plot = pd.DataFrame({
                        "Energy [keV]": energy,
                        "Counts": counts
                    })
                    st.line_chart(df_plot.set_index("Energy [keV]"))
                else:
                    st.warning("Keine Zählungen gefunden.")
    
            except Exception as e:
                st.error(f"Fehler beim Parsen der SPX-Datei: {e}")
    
            finally:
                # Temporäre Datei löschen
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ------------------ Bruker XMethod XADF Viewer ------------------
    elif app_mode == "Bruker XMethod XADF Viewer":
        st.title("🧪 Bruker XMethod XADF Viewer")

        uploaded = st.file_uploader("Upload a .xadf file", type=["xadf","XADF","txt"])
        debug_mode = st.checkbox("🔍 Enable debug mode", value=False)

        if uploaded:
            try:
                tree = ET.parse(uploaded)
                root = tree.getroot()
                parsed = parse_xadf_safe(root)

                sample_name = parsed.get("Info", {}).get("Info", {}).get("APLName", "Unknown Sample")
                st.header(f"📄 Sample: {sample_name}")

                # --- Measurement Conditions ---
                mp = parsed.get("MeasurementParameters", {})
                with st.expander("⚙️ Measurement Conditions", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tube Element:** {mp.get('TubeElement','?')} (Z={mp.get('TubeZ','?')})")
                        st.write(f"**Voltage (kV):** {mp.get('HV') or mp.get('TubeVoltage','?')}")
                        st.write(f"**Current (µA):** {mp.get('Current') or mp.get('TubeCurrent','?')}")
                        st.write(f"**Measurement Time (s):** {mp.get('Time','?')}")
                    with col2:
                        st.write(f"**Number of Detectors:** {mp.get('NumberOfDetectors','?')}")
                        st.write(f"**Spot Size:** {mp.get('Collimator',{}).get('Description','?')}")
                        st.write(f"**Atmosphere:** {mp.get('AtmosphereName','?')}")
                        st.write(f"**Primary Spectrum (PrimarySpc):** {mp.get('PrimarySpc','?')}")

                # --- Layer Composition ---
                st.subheader("🧱 Layer Structure")
                for L in parsed.get("Layers", []):
                    with st.expander(f"Layer {L['Index']}: {L.get('Description','')}", expanded=False):
                        st.write(f"**Thickness:** {L.get('Thickness_um','?')} µm")
                        st.write(f"**Density:** {L.get('Density_gcm3','?')} g/cm³")
                        if L.get("Elements"):
                            df_layer = pd.DataFrame([{
                                "Symbol": e.get("Symbol", "?"),
                                "Conc": e.get("Conc", "?"),
                                "PE_Spc_Number": e.get("PE_Spc_Number", "?"),
                                "Emission Lines": ", ".join(e.get("Lines", []))
                            } for e in L["Elements"]])
                            st.dataframe(df_layer, use_container_width=True)

                # --- Layer Composition Plot ---
                rows = []
                for L in parsed.get("Layers", []):
                    desc = L.get("Description", f"Layer {L['Index']}")
                    for e in L.get("Elements", []):
                        try:
                            conc = float(e.get("Conc", 0) or 0)
                        except:
                            conc = 0
                        rows.append({"Layer": desc, "Element": e.get("Symbol", "?"), "Conc": conc})

                df_plot = pd.DataFrame(rows)
                if not df_plot.empty:
                    normalize = st.checkbox("Normalize layer concentrations", value=False)
                    if normalize:
                        df_plot["Conc_norm"] = df_plot.groupby("Layer")["Conc"].transform(
                            lambda x: x / x.sum() * 100 if x.sum() != 0 else x
                        )
                        yfield = "Conc_norm"
                        ytitle = "Normalized Concentration (%)"
                    else:
                        yfield = "Conc"
                        ytitle = "Concentration"

                    fig = px.bar(
                        df_plot,
                        x="Layer",
                        y=yfield,
                        color="Element",
                        barmode="stack",
                        title="Layer Composition",
                        text_auto=True
                    )
                    fig.update_yaxes(title=ytitle)
                    st.plotly_chart(fig, use_container_width=True)

                # --- JSON Download ---
                full_json = json.dumps(parsed, indent=2, ensure_ascii=False)
                st.download_button(
                    "💾 Save Full JSON",
                    data=full_json,
                    file_name=f"{sample_name}_full.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"❌ Error while parsing file: {e}")
                if debug_mode:
                    uploaded.seek(0)
                    raw_xml = uploaded.read().decode("utf-8", errors="ignore")
                    st.code(raw_xml[:5000] + "\n... (truncated)", language="xml")

if __name__ == "__main__":

    main()
