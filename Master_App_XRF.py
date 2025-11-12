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
            show_plot = st.checkbox("Show spectrum plot", value=True)
            show_json = st.checkbox("Show parsed JSON", value=False)
    
            if uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".spx") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
    
                try:
                    rec = parse_spx_file(tmp_path)
                    st.success("✅ File successfully parsed!")
    
                    # Key info
                    acq = rec.get("acquisition", {})
                    st.subheader("Acquisition Info")
                    st.write(f"**Spectrum:** {rec.get('spectrum_name','–')}")
                    st.write(f"**Real time:** {acq.get('real_time_ms','–')} ms")
                    st.write(f"**Live time:** {acq.get('live_time_ms','–')} ms")
                    st.write(f"**Dead time:** {acq.get('dead_time_percent','–')} %")
    
                    # Spectrum plot
                    if show_plot:
                        counts = np.array(rec.get("counts") or [], dtype=float)
                        if counts.size > 0:
                            x = energy_axis_keV(rec)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(x, counts, lw=1)
                            ax.set_xlabel("Energy (keV)")
                            ax.set_ylabel("Counts")
                            ax.grid(True, alpha=0.3)
                            ax.set_title(rec["spectrum_name"])
                            st.pyplot(fig)
                        else:
                            st.warning("No counts data found.")
    
                    # JSON export
                    json_bytes = json.dumps(rec, indent=2, ensure_ascii=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download parsed JSON",
                        data=json_bytes,
                        file_name=f"{rec['spectrum_name']}_parsed.json",
                        mime="application/json"
                    )
    
                    if show_json:
                        st.json(rec)
    
                except Exception as e:
                    st.error(f"❌ Error parsing file: {e}")

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

