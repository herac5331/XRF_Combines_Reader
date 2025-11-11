# xrf_viewer.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import tempfile
import os
from pymongo import MongoClient

from constants import MOLAR_MASSES
from spx_parser import parse_spx_file, energy_axis_keV
from xadf_parser import parse_xadf_safe


# ==================== XRF VIEWER CLASS ====================
class XRFViewer:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="XRF", collection_name="XRF Data"):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        self.connect_to_database()

    # ---------- FILE & TIME ----------
    def load_file_datetime(self, uploaded_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            ts = os.path.getmtime(tmp_path)
            dt = datetime.fromtimestamp(ts)
            os.unlink(tmp_path)
            return dt
        except Exception as e:
            st.warning(f"⚠️ Could not read Date-Time: {e}")
            return datetime.now()

    # ---------- DATABASE ----------
    def connect_to_database(self):
        try:
            client = MongoClient(self.mongo_uri)
            self.collection = client[self.db_name][self.collection_name]
        except Exception as e:
            st.error(f"MongoDB connection failed: {e}")
            self.collection = None

    def test_connection(self):
        if self.collection is not None:
            try:
                count = self.collection.count_documents({})
                return True, count
            except:
                return False, 0
        return False, 0

    # ---------- FILE PROCESSING ----------
    def load_xrf_file(self, uploaded_file, n_x, n_y, z_value=5):
        try:
            df_raw = pd.read_excel(uploaded_file, header=1)
            first_col = df_raw.iloc[:, 0].astype(str)
            footer_start_idx = next(
                (i for i, val in enumerate(first_col) if any(k in val.lower() for k in ["mean", "std", "average"])),
                None,
            )

            if footer_start_idx is not None:
                df_points = df_raw.iloc[:footer_start_idx].copy()
                df_footer = df_raw.iloc[footer_start_idx:].copy()
            else:
                df_points = df_raw.copy()
                df_footer = pd.DataFrame()

            df_points = df_points.iloc[::-1].reset_index(drop=True)
            n_points = len(df_points)
            expected = n_x * n_y
            if n_points != expected:
                st.warning(f"⚠️ Points: {n_points} ≠ Expected grid: {expected}")

            x_vals = np.linspace(5, 45, n_x)
            y_vals = np.linspace(5, 45, n_y)
            coords = [(x, y, z_value) for y in y_vals for x in x_vals]
            coords_df = pd.DataFrame(coords, columns=["X-position (mm)", "Y-position (mm)", "Z-position (mm)"])
            df_points = pd.concat([coords_df, df_points], axis=1)

            if not df_footer.empty:
                for col in ["X-position (mm)", "Y-position (mm)", "Z-position (mm)"]:
                    df_footer.insert(0, col, np.nan)

            return pd.concat([df_points, df_footer], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    # ---------- CONVERSIONS ----------
    def weight_to_atomic(self, df, elements):
        atom_fracs = pd.DataFrame(index=df.index)
        valid_elements = [el for el in elements if el in MOLAR_MASSES]

        if not valid_elements:
            st.warning("No valid elements found for atomic conversion.")
            return atom_fracs

        numerators = {el: df[f"{el} [%]"] / MOLAR_MASSES[el] for el in valid_elements}
        denom = sum(numerators.values())
        denom.replace(0, np.nan, inplace=True)

        for el in valid_elements:
            atom_fracs[f"{el} [at%]"] = (numerators[el] / denom) * 100

        for el in elements:
            if el not in MOLAR_MASSES:
                st.warning(f"⚠️ No molar mass found for {el}")

        return atom_fracs

    def add_ratio(self, df, ratio_choice):
        if not ratio_choice:
            return df

        el1, el2 = ratio_choice.split("/")

        # Finding Layers with at% suffix 
        cols_el1 = [c for c in df.columns if c.startswith(f"{el1} [at%]")]
        cols_el2 = [c for c in df.columns if c.startswith(f"{el2} [at%]")]

        if not cols_el1 or not cols_el2:
            st.warning(f"⚠️ Missing atomic columns for ratio: {ratio_choice}")
            return df

        # Match same layers with each other
        for c1 in cols_el1:
            layer_tag = ""
            if "(Layer" in c1:
                layer_tag = c1[c1.find("(Layer"): c1.find(")") + 1]
                matching = [c2 for c2 in cols_el2 if layer_tag in c2]
            else:
                matching = cols_el2

            if not matching:
                continue

            c2 = matching[0]
            ratio_col = f"Ratio({ratio_choice}) {layer_tag}".strip()
            df[ratio_col] = df[c1] / df[c2].replace(0, np.nan)

        return df
    

    # ---------- ELEMENT ADJUSTMENT ----------
    def apply_element_correction(self, df, layer_weight_cols, layer_name=""):
        """
        Optional: Adjusts weight-% for a layer using user-defined correction factors,
        then recalculates atomic-% and shows both input and effective factors.
        """
        st.subheader(f"Element Correction Factors — {layer_name}")

        # extract elements
        elements = [c.replace(" [%]", "") for c in layer_weight_cols]

        #correction options with expander
        with st.expander(f"🔧 Advanced Correction Options for {layer_name} (optional)", expanded=False):
            enable_correction = st.checkbox(
                f"Enable manual element correction ({layer_name})",
                value=False,
                key=f"enable_corr_{layer_name}"
            )

            correction_factors = {}
            fixed_elements = {}

            if enable_correction:
                cols = st.columns(len(elements))
                for i, el in enumerate(elements):
                    with cols[i]:
                        correction_factors[el] = st.number_input(
                            f"{el} correction factor",
                            min_value=0.0,
                            value=1.0,
                            step=0.01,
                            key=f"{layer_name}_{el}_factor"
                        )
                        fixed_elements[el] = st.checkbox(
                            f"{el} fixed",
                            value=False,
                            key=f"{layer_name}_{el}_fixed"
                        )
            else:
                # if no correction is applied all facotors are 1 
                correction_factors = {el: 1.0 for el in elements}
                fixed_elements = {el: False for el in elements}

        # Prepare data frame 
        df_corrected = df.copy()
        weights = df[layer_weight_cols].copy()

      
        for idx, row in weights.iterrows():
            adjusted = {}

            # Apply correction factor
            for el, col in zip(elements, layer_weight_cols):
                if fixed_elements[el]:
                    adjusted[el] = row[col]  # fixiert
                else:
                    adjusted[el] = row[col] * correction_factors[el]

            # Norming all elements to 100%
            fixed_sum = sum(adjusted[el] for el in elements if fixed_elements[el])
            non_fixed_sum = sum(adjusted[el] for el in elements if not fixed_elements[el])

            if non_fixed_sum > 0:
                scale_factor = (100 - fixed_sum) / non_fixed_sum
                for el in elements:
                    if not fixed_elements[el]:
                        adjusted[el] *= scale_factor

        
            for el, col in zip(elements, layer_weight_cols):
                df_corrected.at[idx, col] = adjusted[el]

        # Addtitional auto correction factors to check matching
        if enable_correction:
            auto_factors = {}
            for el, col in zip(elements, layer_weight_cols):
                old_mean = df[col].mean()
                new_mean = df_corrected[col].mean()
                auto_factors[el] = round(new_mean / old_mean, 3) if old_mean else np.nan

            st.write(f"✅ Correction Summary for {layer_name}")
            correction_summary = {
                el: {
                    "input_factor": round(correction_factors[el], 3),
                    "effective_factor": auto_factors[el],
                    "fixed": fixed_elements[el]
                }
                for el in elements
            }
            st.json(correction_summary)
        else:
            st.info(f"No manual correction applied for {layer_name} — all elements remain unchanged.")

        return df_corrected
    

    def render_ratio_heatmap(self, df, ratio_choice):

        st.subheader("📊 Ratio Heatmap Visualization")

        # Detects columns automatically
        x_col = next((c for c in df.columns if c.lower().startswith("x")), None)
        y_col = next((c for c in df.columns if c.lower().startswith("y")), None)
        ratio_col = next((c for c in df.columns if ratio_choice in c), None)

        if not x_col or not y_col or not ratio_col:
            st.warning(f"⚠️ Could not generate heatmap — missing coordinate or ratio columns.\n"
                    f"Found X='{x_col}', Y='{y_col}', Ratio='{ratio_col}'")
            return

        try:
            
            df_plot = df[[x_col, y_col, ratio_col]].dropna()
            df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors="coerce")
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors="coerce")
            df_plot[ratio_col] = pd.to_numeric(df_plot[ratio_col], errors="coerce")
            df_plot = df_plot.dropna()

            if df_plot.empty:
                st.warning("⚠️ No valid data available for plotting.")
                return

            # Color skale
            ratio_min, ratio_max = float(df_plot[ratio_col].min()), float(df_plot[ratio_col].max())
            st.caption(f"Detected ratio range: **{ratio_min:.4f} – {ratio_max:.4f}**")

            min_val, max_val = st.slider(
                "Color range (min–max)",
                min_value=float(ratio_min),
                max_value=float(ratio_max),
                value=(float(ratio_min), float(ratio_max)),
                step=(ratio_max - ratio_min) / 100 if ratio_max > ratio_min else 0.01,
            )

            # Color map
            colormaps = [
                "viridis", "plasma", "inferno", "magma", "cividis",
                "coolwarm", "Spectral", "YlGnBu", "turbo"
            ]
            selected_cmap = st.selectbox("Color map", colormaps, index=0)

        
            pivot_table = df_plot.pivot_table(index=y_col, columns=x_col, values=ratio_col, aggfunc="mean")

            # Figure
            fig, ax = plt.subplots(figsize=(5, 4))
            cax = ax.imshow(
                pivot_table.values,
                cmap=selected_cmap,
                origin="lower",
                extent=[
                    pivot_table.columns.min(),
                    pivot_table.columns.max(),
                    pivot_table.index.min(),
                    pivot_table.index.max(),
                ],
                aspect="auto",
                vmin=min_val,
                vmax=max_val
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{ratio_choice} Ratio Heatmap", fontsize=11)
            fig.colorbar(cax, ax=ax, label=ratio_choice)

            # Display
            st.pyplot(fig, use_container_width=False)

            # Export
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
            st.download_button(
                "💾 Download Heatmap as PNG",
                data=buf.getvalue(),
                file_name=f"heatmap_{ratio_choice.replace('/', '_')}.png",
                mime="image/png",
            )

        except Exception as e:
            st.error(f"❌ Heatmap generation failed: {e}")



    # ---------- UPLOAD TAB ----------
    def render_upload_tab(self):
        st.header("Upload XRF File")
        uploaded_file = st.file_uploader("Choose XLS/XLSX file", type=["xls", "xlsx"])

        all_atomic_cols = []

        # === Sample Info ===
        st.subheader("Sample Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.sample_id = st.text_input("Sample ID", st.session_state.get("sample_id", ""))
        with col2:
            st.session_state.institution = st.text_input("Institution", st.session_state.get("institution", ""))
        with col3:
            st.session_state.operator = st.text_input("Operator", st.session_state.get("operator", ""))

        treat_methods = ["As-deposited", "Annealing", "UV-Ozone", "Irradiation", "Plasma"]
        treat_method = st.selectbox("Treatment Method", treat_methods)
        treat_sequence = 0 if treat_method == "As-deposited" else st.number_input(
            "Treatment Sequence", 1, step=1, value=1
        )

        if uploaded_file:
            # === Grid Parameter ===
            st.subheader("Measurement Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                n_x = st.number_input("Measurement Points in X-Direction", min_value=1, value=10, step=1)
            with col2:
                n_y = st.number_input("Measurement Points in Y-Direction", min_value=1, value=10, step=1)
            with col3:
                z_val = st.number_input("Z-Coordinate (mm)", min_value=0.0, value=5.0, step=0.5)

            # === Date/Time from file ===
            file_dt = self.load_file_datetime(uploaded_file)

            # === User can adjust Date/Time ===
            col_date, col_time = st.columns(2)
            with col_date:
                selected_date = st.date_input("Select Date", value=file_dt.date())
            with col_time:
                selected_time = st.time_input("Select Time", value=file_dt.time())

            # === Load and prepare data ===
            df = self.load_xrf_file(uploaded_file, n_x, n_y, z_val)
            if df is None:
                st.stop()

            dt_combined = datetime.combine(selected_date, selected_time)
            df.insert(3, "Date", dt_combined.strftime("%m/%d/%Y"))
            df.insert(4, "Time", dt_combined.strftime("%H:%M:%S"))

            # ============================================================
            # === Deconvolution Settings
            # ============================================================
            st.subheader("Deconvolution Settings")
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox("Deconvolution Method (only important for MongoDB)", [
                    "Bayes deconvolution",
                    "Bayes series deconvolution",
                    "Bayes profile deconvolution",
                    "Fit",
                    "Fit series deconvolution",
                    "Fit group deconvolution"
                ])
            with col2:
                xmethod_name = st.text_input("XMethod Name", placeholder="e.g. CsPbI3_SeriesFit_Henry_Aug2025")

            # ============================================================
            # === Layer-based element handling and correction ============
            # ============================================================
            thickness_cols = [col for col in df.columns if "Thickn" in col]
            layer_boundaries = [df.columns.get_loc(c) for c in thickness_cols]

            if not layer_boundaries:
                layer_boundaries = [4]  # after Date/Time
            layer_boundaries.append(len(df.columns))  # mark end

            layer_info = {} # saves layer: {elements, weight_cols, atomic_cols}

            for i, start_idx in enumerate(layer_boundaries[:-1]):
                end_idx = layer_boundaries[i + 1]
                layer_cols = df.columns[start_idx + 1:end_idx]
                elements = [c.replace(" [%]", "") for c in layer_cols if c.endswith("[%]")]
                if not elements:
                    continue

                st.write(f"🔹 Processing Layer {i + 1} ({', '.join(elements)})")

                # ✅ Apply correction
                df = self.apply_element_correction(df, layer_cols, f"Layer_{i+1}")

                # ✅ Recalculates atomic percentage with corrected values
                weight_df_corrected = df[layer_cols].copy()
                atomic_df = self.weight_to_atomic(weight_df_corrected, elements)

                atomic_cols_layer = []
                for col in atomic_df.columns:
                    new_col = f"{col} (Layer {i + 1})"
                    df[new_col] = atomic_df[col]
                    atomic_cols_layer.append(new_col)
                    all_atomic_cols.append(new_col)

                layer_info[f"Layer {i + 1}"] = {
                    "elements": elements,
                    "weight_cols": layer_cols,
                    "atomic_cols": atomic_cols_layer
                }


            # ============================================================
            # === Ratio Handling + Heatmap Visualization ================
            # ============================================================
            atomic_elements = [c.split(" [at%]")[0] for c in all_atomic_cols]
            ratio_options = [f"{a}/{b}" for a in atomic_elements for b in atomic_elements if a != b]
            ratio_choice = st.selectbox("Select Ratio", [""] + ratio_options)

            if ratio_choice:
                df = self.add_ratio(df, ratio_choice)

                # Heatmap expander
                with st.expander("📈 Show Ratio Heatmap", expanded=True):
                    self.render_ratio_heatmap(df, ratio_choice)
                    

            # ============================================================
            # === Display + Export + Upload
            # ============================================================
            st.subheader("Processed Data")
            st.dataframe(df, use_container_width=True, height=500)

            self.export_csv_with_datetime(df, dt_combined, treat_method, treat_sequence)

            if st.button("Upload to MongoDB", type="primary"):
                success, msg = self.save_to_mongo(df, uploaded_file.name, method, xmethod_name)
                st.success(msg) if success else st.error(msg)


    # ---------- SEARCH TAB ----------
    def render_search_tab(self):
        st.header("Search Data Results")
        if self.collection is None:
            st.warning("No MongoDB connection")
            return

        query = {}
        sample = st.text_input("Search by filename:")
        method = st.selectbox("Deconvolution method:", [
            "", "Bayes deconvolution", "Bayes series deconvolution", "Bayes profile deconvolution",
            "Fit", "Fit series deconvolution", "Fit group deconvolution"
        ])

        if sample:
            query["filename"] = {"$regex": sample, "$options": "i"}
        if method:
            query["deconvolution_method"] = method

        if query:
            results = list(self.collection.find(query))
            if results:
                df = pd.DataFrame(results)
                st.success(f"Found {len(results)} records")
                st.dataframe(df, use_container_width=True, height=500)
            else:
                st.warning("No results found")

    # ---------- EXPORT ----------
    def export_csv_with_datetime(self, df, dt, treat_method, treat_sequence):
        try:
            df_export = df.copy()
            cols_to_drop = df_export.columns[5:8]
            df_export.drop(columns=cols_to_drop, inplace=True, errors="ignore")

            if len(df_export) > 3:
                df_export = df_export.iloc[:-3]

            csv_text = df_export.to_csv(index=False)
            timestamp = dt.strftime("%Y%m%d_%H%M%S")
            csv_filename = (
                f"{st.session_state.sample_id}_{st.session_state.institution}_{st.session_state.operator}_"
                f"{treat_method}_{treat_sequence}_xrf_mapping_{timestamp}.csv"
            )
            st.download_button("📥 Export CSV", csv_text, file_name=csv_filename, mime="text/csv")
        except Exception as e:
            st.error(f"CSV Export failed: {e}")


#----------Save to Mongo DB-----------#

    def save_to_mongo(self, df, filename, method, xmethod_name):
        """
        Save processed XRF data and metadata to MongoDB.
        """
        if self.collection is None:
            return False, "❌ No MongoDB connection."

        try:
            # --- Prepare metadata ---
            record = {
                "filename": filename,
                "sample_id": st.session_state.get("sample_id", ""),
                "institution": st.session_state.get("institution", ""),
                "operator": st.session_state.get("operator", ""),
                "treatment_method": st.session_state.get("treatment_method", ""),
                "treatment_sequence": st.session_state.get("treatment_sequence", ""),
                "deconvolution_method": method,
                "xmethod_name": xmethod_name,
                "upload_time": datetime.now(),
            }

            # --- Prepare data ---
            # Keep numeric and atomic data clean
            record["data"] = df.to_dict(orient="records")

            # --- Insert into MongoDB ---
            self.collection.insert_one(record)

            return True, f"✅ Data successfully uploaded to MongoDB ({method}, {xmethod_name})"

        except Exception as e:
            return False, f"❌ Upload failed: {e}"


    # ---------- MAIN UI ----------
    def run_streamlit_app(self):
        st.set_page_config(page_title="XRF Viewer", page_icon="🧪", layout="wide")
        st.title("🧪 XRF Data Viewer")
        tab1, tab2 = st.tabs(["Upload XRF File", "Search Data Results"])
        with tab1:
            self.render_upload_tab()
        with tab2:
            self.render_search_tab()


# ==================== MAIN ====================
# def main():
#     viewer = XRFViewer()
#     viewer.run_streamlit_app()


# if __name__ == "__main__":
#     main()
