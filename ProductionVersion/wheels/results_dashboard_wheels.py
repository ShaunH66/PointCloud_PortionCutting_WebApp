# dashboard_cheese_wheel.py

import streamlit as st
import pandas as pd
import pickle
import os
import time

# --- Import Custom Libraries ---
try:
    from functionlib import (
        plot_3d_wheel_with_cuts,
        plot_angular_weight_profile,
        launch_o3d_viewer_with_wedge_cuts,
        _open3d_installed
    )
except ImportError as e:
    st.error(f"FATAL ERROR: Could not import from library file: {e}")
    st.stop()

# --- Configuration ---
DEFAULT_PAYLOAD_FILE = "latest_run_payload_wheel.pkl"
ARCHIVE_BASE_FOLDER = "run_archives_wheel"
AUTO_REFRESH_INTERVAL_SECONDS = 5

# --- App Layout ---
st.set_page_config(
    layout="wide", page_title="Cheese Wheel Portioning Viewer", page_icon="🧀")


def load_payload_from_source(file_source, source_id):
    try:
        if st.session_state.get('last_loaded_file_id') != source_id:
            payload = pickle.load(file_source)
            st.session_state.payload = payload
            st.session_state.last_loaded_file_id = source_id
            st.toast("Wedge results loaded!", icon="🧀")
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load or parse payload file: {e}")


@st.cache_data(ttl=10)
def get_recent_runs(archive_folder):
    recent_runs = []
    if os.path.isdir(archive_folder):
        for filename in os.listdir(archive_folder):
            if filename.startswith("payload_") and filename.endswith(".pkl"):
                file_path = os.path.join(archive_folder, filename)
                try:
                    mod_time = os.path.getmtime(file_path)
                    recent_runs.append((filename, file_path, mod_time))
                except OSError:
                    continue
    recent_runs.sort(key=lambda x: x[2], reverse=True)
    return recent_runs


# --- Main App Body ---
st.title("🧀 Cheese Wheel Portioning Dashboard")
st.markdown("---")

# --- Initialize Session State ---
if 'last_loaded_file_id' not in st.session_state:
    st.session_state.last_loaded_file_id = None
if 'auto_load_enabled' not in st.session_state:
    st.session_state.auto_load_enabled = False

# Sidebar Controls...
with st.sidebar:
    st.header("⚙️ Controls")
    st.toggle("Auto-Load Latest Run",
              value=st.session_state.auto_load_enabled, key="auto_load_enabled")
    if st.button("🔄 Load Latest Run", use_container_width=True, type="primary", disabled=st.session_state.auto_load_enabled):
        if os.path.exists(DEFAULT_PAYLOAD_FILE):
            with open(DEFAULT_PAYLOAD_FILE, "rb") as f:
                load_payload_from_source(
                    f, f"{DEFAULT_PAYLOAD_FILE}-{os.path.getmtime(f.fileno())}")
    st.markdown("---")
    st.subheader("📁 Recent Runs")
    col1, col2 = st.columns([3, 2])
    col1.button("🔄 Refresh List", use_container_width=True,
                on_click=get_recent_runs.clear)
    col2.number_input("Show:", 5, 100, 15, key="num_runs_to_show")
    recent_runs_list = get_recent_runs(ARCHIVE_BASE_FOLDER)
    if recent_runs_list:
        run_options = {os.path.splitext(fname)[
            0]: fpath for fname, fpath, _ in recent_runs_list[:st.session_state.num_runs_to_show]}
        selected_run_name = st.selectbox("Select a past run:", options=list(
            run_options.keys()), index=None, placeholder="Choose an archive...")
        if selected_run_name:
            with open(run_options[selected_run_name], "rb") as f:
                load_payload_from_source(
                    f, f"{run_options[selected_run_name]}-{os.path.getmtime(f.fileno())}")
    st.markdown("---")
    if "payload" in st.session_state:
        st.subheader("Interactive 3D Viewer")
        if _open3d_installed and st.button("👁️ View with Cuts (Open3D)", use_container_width=True):
            payload = st.session_state.payload
            display_cloud = payload.get("processed_point_cloud_for_display_df")
            portions = payload.get("calculation_results",
                                   {}).get("portions", [])
            if display_cloud is not None and not display_cloud.empty and portions:
                launch_o3d_viewer_with_wedge_cuts(display_cloud, portions)
        st.markdown("---")
        if st.button("🧹 Clear Displayed Data", use_container_width=True):
            del st.session_state.payload
            st.session_state.last_loaded_file_id = None
            st.rerun()

# --- Main Content Area ---
if "payload" not in st.session_state:
    st.info("👈 Load a run from the sidebar to begin.")
else:
    payload = st.session_state.payload
    display_cloud = payload.get("processed_point_cloud_for_display_df")
    calc_results = payload.get("calculation_results")
    params = payload.get("input_params_summary_for_ui", {})

    # KPI section...
    st.header("📈 Run Overview")
    if calc_results:
        portions = calc_results.get('portions', [])
        total_weight = params.get('total_weight', 0)
        is_redistribution_mode = params.get('waste_redistribution', False)
        strategy = "Enabled" if is_redistribution_mode else "Disabled"
        if is_redistribution_mode:
            good_wedges = portions
            balance_weight = 0.0
            yield_percent = 100.0
            avg_wedge_weight = sum(
                p['Weight (g)'] for p in good_wedges) / len(good_wedges) if good_wedges else 0
            balance_display = "N/A"
            good_wedges_display = f"{len(good_wedges)} (Equal)"
        else:
            balance_wedge = next(
                (p for p in portions if str(p['Portion #']) == 'Balance'), None)
            balance_weight = balance_wedge['Weight (g)'] if balance_wedge else 0.0
            good_wedges = [p for p in portions if str(
                p['Portion #']) != 'Balance']
            yield_percent = (total_weight - balance_weight) / \
                total_weight * 100 if total_weight > 0 else 0
            avg_wedge_weight = sum(
                p['Weight (g)'] for p in good_wedges) / len(good_wedges) if good_wedges else 0
            balance_display = f"{balance_weight:.2f} g"
            good_wedges_display = f"{len(good_wedges)}"
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("✅ Yield", f"{yield_percent:.2f} %")
        kpi_cols[1].metric("⚖️ Avg. Wedge Weight", f"{avg_wedge_weight:.2f} g")
        kpi_cols[2].metric("#️⃣ Total Good Wedges", good_wedges_display)
        kpi_cols[3].metric("🗑️ Balance / Waste Weight", balance_display)
        kpi_cols[4].metric("⚖️ Total Weight", f"{total_weight:.2f} g")
        kpi_cols[5].metric(label="♻️ Waste Redistribution", value=strategy)

    st.markdown("---")

    tab_summary, tab_details, tab_help = st.tabs(
        ["📊 **Summary & 3D View**", "⚙️ **Run Details**", "ℹ️ **Help & Process Info**"])

    with tab_summary:
        col_3d, col_table = st.columns([6, 4])

        with col_3d:
            st.subheader("3D View & Weight Distribution")
            if display_cloud is not None and not display_cloud.empty and calc_results:
                portions = calc_results.get('portions', [])
                highlighted_num = st.session_state.get(
                    'highlighted_portion')  # Read selection

                fig_3d = plot_3d_wheel_with_cuts(
                    display_cloud, portions, highlighted_portion_num=highlighted_num)
                st.plotly_chart(fig_3d, use_container_width=True)

                total_weight = params.get('total_weight', 0)
                fig_polar = plot_angular_weight_profile(
                    params, total_usable_weight=total_weight)
                st.plotly_chart(fig_polar, use_container_width=True)

        with col_table:
            st.subheader("Wedge Cut Results")
            if calc_results and calc_results.get("portions"):
                portions = calc_results.get("portions")
                res_df = pd.DataFrame(portions)
                if 'Portion #' in res_df.columns:
                    res_df['Portion #'] = res_df['Portion #'].astype(str)

                radio_options = ["None"] + list(res_df["Portion #"])
                st.radio(
                    "**Select a Portion to Highlight:**",
                    options=radio_options,
                    key='highlighted_portion',
                    horizontal=True,
                )

                st.dataframe(
                    res_df.style.format({
                        'Start Angle (deg)': '{:.2f}',
                        'End Angle (deg)': '{:.2f}',
                        'Incremental Angle (deg)': '{:.2f}',
                        'Weight (g)': '{:.2f}'
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                st.error("No portion data found.")

    with tab_details:
        st.header("Detailed Run Information")
        st.subheader("Input Parameters")
        st.json(params, expanded=True)
        st.subheader("Processing Log")
        pipeline_log = payload.get("pipeline_log", [])
        st.text_area("Log", "".join(pipeline_log), height=400, disabled=True)

    with tab_help:
        st.header("ℹ️ Understanding the Dashboard & Process")
        st.markdown("---")

        st.subheader("The Calculation Process")
        st.markdown("""
        The software calculates the wedge cutting plan in a two-stage process to ensure accuracy.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Part 1: Measuring Total Volume & Density")
            st.markdown("""
            1.  **Horizontal Slicing:** The 3D point cloud is sliced into thin horizontal layers from bottom to top.
            2.  **Area Calculation:** The area of each slice is calculated.
            3.  **Volume Summation:** The volume of all slices are added together to get the wheel's total volume.
            4.  **Density Calculation:** The final, crucial density is calculated using the user-provided total weight:
                `Density = Total Weight / Total Volume`
            """)
        # with col2:
        #           Placeholder for an image..
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Part 2: Planning the Wedge Cuts")
            st.markdown("""
            Once the density is known, the software plans the cuts.
            1.  **Find Center:** The geometric center of the wheel is found.
            2.  **Angular Profiling:** The wheel is divided into many thin angular sectors (like spokes). The weight of each sector is calculated based on the density of scanned points within it.
            3.  **Cumulative Profile:** The software creates a profile of the total accumulated weight as it moves around the wheel.
            4.  **Determine Cuts:** It uses this profile to find the exact angle needed to achieve the target weight for each wedge. This automatically compensates for any irregularities in the wheel's shape.
            """)
        # with col4:
        #           Placeholder for an image..)

        st.markdown("---")

        st.subheader("Understanding the Plots")

        with st.expander("📊 **3D View & Weight Distribution**"):
            st.markdown("""
            - **3D View:** This shows a top-down view of the scanned point cloud.
                - **Red Lines:** Indicate the position of each cut.
                - **Green Arrow:** Shows the starting position and direction of the **first cut**.
                - **Blue Curved Arrow:** Shows the direction the turntable will rotate to make subsequent cuts (counter-clockwise).
                - **Cyan Highlight:** When you select a portion using the radio buttons, the points belonging to that wedge are highlighted in cyan.
            - **Angular Weight Distribution Profile:** This polar chart is a "fingerprint" of the wheel.
                - A point **further from the center** means that angular slice of the wheel is **heavier**.
                - A point **closer to the center** means that slice is **lighter**.
                - This plot explains why the `Incremental Angle` for each wedge might be different; the software cuts a wider angle in lighter areas and a narrower angle in heavier areas to achieve a consistent weight.
            """)

        with st.expander("⚙️ **Key Calculation Parameters**"):
            st.markdown("""
            These parameters, set in the `main_cheese_wheel.py` script, control the calculation logic.
            - **`waste_redistribution`**:
                - `True`: The system calculates a new, optimal target weight to make all portions equal, maximizing yield. There will be no "Balance" wedge.
                - `False`: The system uses the exact `target_weight` and creates a final, smaller "Balance" wedge with the leftover material.
            - **`guarantee_overweight`**:
                - `True`: (Only works when `waste_redistribution` is `False`). Ensures every portion is *at least* the target weight by rounding up the cut. This is for meeting minimum weight specs for customers.
                - `False`: The system uses interpolation to get as close to the target weight as possible, which may result in some portions being slightly under.
            - **`num_angular_slices`**:
                - This controls the precision of the weight profile. A higher number (e.g., `3600`) results in smaller angular slices, allowing the `guarantee_overweight` feature to be more accurate (less giveaway).
            """)

# Auto-refresh logic...
if st.session_state.auto_load_enabled:
    if os.path.exists(DEFAULT_PAYLOAD_FILE):
        try:
            current_mod_time = os.path.getmtime(DEFAULT_PAYLOAD_FILE)
            current_file_id = f"{DEFAULT_PAYLOAD_FILE}-{current_mod_time}"
            if st.session_state.last_loaded_file_id != current_file_id:
                with open(DEFAULT_PAYLOAD_FILE, "rb") as f:
                    load_payload_from_source(f, current_file_id)
        except Exception:
            pass
    time.sleep(AUTO_REFRESH_INTERVAL_SECONDS)
    st.rerun()
