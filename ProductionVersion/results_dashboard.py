import numpy as np
import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import sys
import time

try:
    from functionlib import (
        plot_3d_loaf,
        plot_area_profile,
        plot_cumulative_weight,
        calculate_slice_profile,
        start_o3d_visualization,
        launch_o3d_viewer_with_cuts,
        _open3d_installed,
        FLOAT_EPSILON
    )
except ImportError as e:
    st.error(f"FATAL ERROR: Could not import from library file: {e}")
    st.stop()

# --- Configuration ---
DEFAULT_PAYLOAD_FILE = "latest_run_payload.pkl"
ARCHIVE_BASE_FOLDER = "run_archives"
AUTO_REFRESH_INTERVAL_SECONDS = 5

# --- App Layout ---
st.set_page_config(
    layout="wide", page_title="Portioning Results Viewer", page_icon="📊")

# --- Helper Functions ---
def format_df_for_display(df):
    """Applies number formatting to a results DataFrame."""
    format_mapping = {
        "Start Y (mm)": '{:.2f}',
        "End Y (mm)": '{:.2f}',
        "Length (mm)": '{:.2f}',
        "Weight (g)": '{:.2f}'
    }
    return df.style.format(format_mapping)


def load_payload_from_source(file_source, source_id):
    """Loads a payload and updates session state, preventing re-processing."""
    try:
        if st.session_state.get('last_loaded_file_id') != source_id:
            payload = pickle.load(file_source)
            st.session_state.payload = payload
            st.session_state.last_loaded_file_id = source_id
            st.toast("Results loaded successfully!", icon="✅")
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load or parse payload file: {e}")
        
        
@st.cache_data(ttl=10) # Cache the file list for 10 seconds to avoid constant disk reads
def get_recent_runs(archive_folder):
    """Scans the archive folder and returns a sorted list of recent payload files."""
    recent_runs = []
    if os.path.isdir(archive_folder):
        for filename in os.listdir(archive_folder):
            if filename.startswith("payload_") and filename.endswith(".pkl"):
                file_path = os.path.join(archive_folder, filename)
                try:
                    mod_time = os.path.getmtime(file_path)
                    recent_runs.append((filename, file_path, mod_time))
                except OSError: continue
    recent_runs.sort(key=lambda x: x[2], reverse=True)
    return recent_runs


# --- Main App Body ---
st.title("📊 Point Cloud Portioning Dashboard")
st.markdown("---")

# Initialize state if it's the first run
if 'last_loaded_file_id' not in st.session_state:
    st.session_state.last_loaded_file_id = None

if 'last_loaded_file_id' not in st.session_state:
    st.session_state.last_loaded_file_id = None
if 'auto_load_enabled' not in st.session_state:
    st.session_state.auto_load_enabled = False

if 'num_runs_to_show' not in st.session_state:
    st.session_state.num_runs_to_show = 15

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    # --- Option 1: Load the latest run ---
    st.toggle(
        "Auto-Load Latest Run", 
        value=st.session_state.auto_load_enabled, 
        key="auto_load_enabled",
        help="When enabled, this app will automatically load new results from 'latest_run_payload.pkl' as it's updated."
    )
    
    # --- Manual Load Button (still useful when auto-load is off) ---
    if st.button("🔄 Load Latest Run", use_container_width=True, type="primary", disabled=st.session_state.auto_load_enabled):
        if os.path.exists(DEFAULT_PAYLOAD_FILE):
            with open(DEFAULT_PAYLOAD_FILE, "rb") as f:
                file_id = f"{DEFAULT_PAYLOAD_FILE}-{os.path.getmtime(DEFAULT_PAYLOAD_FILE)}"
                load_payload_from_source(f, file_id)
        else:
            st.warning(f"Default file not found: '{DEFAULT_PAYLOAD_FILE}'.")

    st.markdown("---")
    
    st.subheader("📁 Recent & Specific Runs")

    # --- Row for Refresh Button and Number Input ---
    col1, col2 = st.columns([3, 2])
    with col1:
        # This button clears the cache, forcing get_recent_runs to re-scan the disk
        if st.button("🔄 Refresh List", use_container_width=True):
            get_recent_runs.clear()
            st.toast("Recent runs list refreshed!")
            st.rerun()
    with col2:
        st.number_input(
            "Show:", 
            min_value=5, 
            max_value=5000, 
            step=5,
            key="num_runs_to_show",
            help="Number of recent runs to display in the dropdown."
        )

    # --- Dropdown for Recent Runs ---
    recent_runs_list = get_recent_runs(ARCHIVE_BASE_FOLDER)
    if not recent_runs_list:
        st.caption("No archived runs found in 'run_archives' folder.")
    else:
        num_to_show = st.session_state.num_runs_to_show
        run_options = {os.path.splitext(fname)[0]: fpath for fname, fpath, _ in recent_runs_list[:num_to_show]}
        
        selected_run_name = st.selectbox(
            "Select a past run to load:", options=list(run_options.keys()),
            index=None, placeholder="Choose from archives..."
        )
        if selected_run_name:
            file_path = run_options[selected_run_name]
            with open(file_path, "rb") as f:
                file_id = f"{file_path}-{os.path.getmtime(file_path)}"
                load_payload_from_source(f, file_id)

    st.markdown("---")
    
    # --- Option 2: Manually upload a specific run file ---
    st.subheader("Load a Specific Run")
    uploaded_file = st.file_uploader(
        "Choose a payload file", 
        type=['pkl'],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        load_payload_from_source(uploaded_file, file_id)

    # --- Open3D Action Buttons ---
    if "payload" in st.session_state:
        st.markdown("---")
        st.subheader("Interactive 3D Viewers\nRequires Open3D Library")
        
        if _open3d_installed:
            payload = st.session_state.payload
            display_cloud = payload.get("processed_point_cloud_for_display_df")
            calc_results = payload.get("calculation_results")

            # Button for Fly-Around View
            if st.button("🚁 Fly Around Cloud", use_container_width=True):
                if display_cloud is not None and not display_cloud.empty:
                    start_o3d_visualization(display_cloud)
                else:
                    st.warning("No point cloud data loaded to visualize.")

            # Button for View with Cuts
            if st.button("👁️ View with Cuts", use_container_width=True):
                if display_cloud is not None and not display_cloud.empty and calc_results:
                    # For Open3D, we must use the original, real-world coordinates
                    original_portions = calc_results.get("portions", [])
                    scanner_offset = calc_results.get('y_offset_for_plot', 0.0)
                    
                    o3d_portions = []
                    for p in original_portions:
                        p_copy = p.copy()
                        p_copy['display_start_y'] += scanner_offset
                        p_copy['display_end_y'] += scanner_offset
                        if 'cut_y' in p_copy:
                            p_copy['cut_y'] += scanner_offset
                        o3d_portions.append(p_copy)
                    
                    launch_o3d_viewer_with_cuts(
                        display_cloud,
                        o3d_portions,
                        calc_results.get("calc_start_y", 0) + scanner_offset,
                        calc_results.get("calc_end_y", 0) + scanner_offset,
                        0
                    )
                else:
                    st.warning("Cannot show cuts: Point cloud or calculation results are missing.")
        else:
            st.warning("Open3D library not installed. 3D viewers are disabled.", icon="⚠️")
    
    st.markdown("---")
       
    # --- Display Status and Clear Button ---
    if "payload" in st.session_state:
        st.success("Data is currently loaded.")
        if st.button("🧹 Clear Displayed Data", use_container_width=True, type="secondary"):
            del st.session_state.payload
            st.session_state.last_loaded_file_id = None # Reset the file tracker
            st.rerun()
    else:
        st.info(f"Waiting for user to load data.")

    st.markdown("---")
        
# --- Main Content ---

if "payload" not in st.session_state:
    st.info("👈 Click 'Load Results from File' in the controls panel to begin.")
else:
    # Extract all necessary data from the loaded payload
    payload = st.session_state.payload
    display_cloud = payload.get("processed_point_cloud_for_display_df")
    df_for_inspector = payload.get("df_for_slice_inspector")
    calc_results = payload.get("calculation_results")
    params = payload.get("input_params_summary_for_ui", {})
    pipeline_log = payload.get("pipeline_log", [])

    # --- Header with Key Performance Indicators (KPIs) ---
    st.header("📈 Run Overview & Key Metrics")
    if calc_results:
        portions = calc_results.get('portions', [])
        good_portions_count = calc_results.get('good_portions_count', 0)
        good_weight_total = calc_results.get('good_weight', 0)
        waste_grams = calc_results.get('waste_weight', 0)
        total_loaf_weight = calc_results.get('total_loaf_weight', 0)
        yield_percent = yield_percent = calc_results.get('yield_percentage', 0)
        avg_portion_weight = good_weight_total / good_portions_count if good_portions_count > 0 else 0
        product_length_mm = calc_results.get('product_length', 0)
        strategy = "Enabled" if params.get('waste_redistribution', False) else "Disabled"
        
        kpi_row1_cols = st.columns(4)
        kpi_row1_cols[0].metric(label="✅ Yield", value=f"{yield_percent:.2f} %")
        kpi_row1_cols[1].metric(label="🗑️ Waste / Balance Portion", value=f"{waste_grams:.2f} g")
        kpi_row1_cols[2].metric(label="📏 Product Length", value=f"{product_length_mm:.2f} mm")
        kpi_row1_cols[3].metric(label="♻️ Waste Redistribution", value=strategy)

        # Add a space between rows
        st.write("") 
        
        kpi_row2_cols = st.columns(4)
        with kpi_row2_cols[0]:
            original_target = params.get('target_weight', 0)
            if strategy == "Enabled":
                optimized_target = calc_results.get('optimized_target_weight')
                if optimized_target is not None:
                    delta_str = f"{optimized_target - original_target:+.2f}g vs Original"
                    st.metric("⚖️ Optimized Target", f"{optimized_target:.2f} g", delta_str, "normal")
                else:
                    st.metric("⚖️ Original Target", f"{original_target:.2f} g", "Opt. limit exceeded", "inverse")          
            else:
                st.metric("🎯 Target Weight", f"{(original_target):.2f} g")
        
        kpi_row2_cols[1].metric("Good Portions", f"{good_portions_count}")
        kpi_row2_cols[2].metric("Avg. Portion Wt.", f"{avg_portion_weight:.2f} g")
        kpi_row2_cols[3].metric("Total Loaf Wt.", f"{total_loaf_weight / 1000:.3f} kg")
        
    st.markdown("---")

    # --- Main Visualization Tabs ---
    tab_summary, tab_inspector, tab_details = st.tabs(
        ["📊 **Summary & 3D View**", "🔬 **Slice Inspector**", "⚙️ **Run Details**"])

    with tab_summary:
        col_3d, col_table = st.columns([6, 4])

        with col_3d:
            st.subheader("3D View of Processed Cloud")
            show_cuts = st.toggle("Show Cut Planes", value=True)
            
            plot_cloud = payload.get("df_for_slice_inspector", display_cloud)
            
            if plot_cloud is not None and not plot_cloud.empty:
                highlighted_num = st.session_state.get('highlighted_stick_portion')

                fig_3d = plot_3d_loaf(
                    plot_cloud,
                    portions=calc_results.get("portions", []),
                    title="",
                    highlighted_portion_num=highlighted_num,
                    show_cut_planes=show_cuts
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("No point cloud data found in the payload to display.")

        with col_table:
            st.subheader("Portioning Results")
            if calc_results and calc_results.get("portions"):
                original_portions = calc_results.get("portions", [])
                scanner_offset = calc_results.get('y_offset_for_plot', 0.0)
                
                display_data = []
                if params.get("enable_y_normalization"):
                    block_start_y_real = original_portions[0]['display_start_y'] + scanner_offset if original_portions else 0
                    for p in original_portions:
                        display_data.append({
                            "Portion #": p['portion_num'],
                            "Start Y (mm)": (p['display_start_y'] + scanner_offset) - block_start_y_real,
                            "End Y (mm)": (p['display_end_y'] + scanner_offset) - block_start_y_real,
                            "Length (mm)": p['length'],
                            "Weight (g)": p['weight']
                        })
                else:
                    for p in original_portions:
                        display_data.append({
                            "Portion #": p['portion_num'],
                            "Start Y (mm)": p['display_start_y'] + scanner_offset,
                            "End Y (mm)": p['display_end_y'] + scanner_offset,
                            "Length (mm)": p['length'],
                            "Weight (g)": p['weight']
                        })
                
                res_df = pd.DataFrame(display_data)
                if 'Portion #' in res_df.columns:
                    res_df['Portion #'] = res_df['Portion #'].astype(str)

                radio_options = ["None"] + list(res_df["Portion #"])
                st.radio(
                    "**Select a Portion to Highlight:**",
                    options=radio_options,
                    key='highlighted_stick_portion',
                    horizontal=True,
                )
                
                st.dataframe(
                    res_df.style.format({
                        'Start Y (mm)': '{:.2f}',
                        'End Y (mm)': '{:.2f}',
                        'Length (mm)': '{:.2f}',
                        'Weight (g)': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("No portion data found in the results.")

    with tab_inspector:
        st.header("Analysis Plots & Slice Inspector")
        if calc_results:
            vp, ys, density = calc_results.get("volume_profile", {}), calc_results.get(
                "sorted_y_starts", np.array([])), calc_results.get("density", 0)
            slice_thickness = params.get(
                "slice_thickness", 0.5)

            if vp and ys.size > 0:
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.subheader("Area Profile")
                    fig_area = plot_area_profile(vp, ys, slice_thickness)
                    st.plotly_chart(fig_area, use_container_width=True)
                with plot_col2:
                    st.subheader("Cumulative Weight")
                    fig_weight = plot_cumulative_weight(
                        vp, ys, density, calc_results["portions"], params.get(
                            "target_weight", 0),
                        params.get("weight_tolerance", 0), slice_increment_mm=slice_thickness
                    )
                    st.plotly_chart(fig_weight, use_container_width=True)

                st.divider()
                st.subheader("🔬 Interactive Slice Inspector")

                if df_for_inspector is not None and not df_for_inspector.empty:
                    ys_calc = calc_results.get(
                        "sorted_y_starts", np.array([]))
                    if ys_calc.size > 0:
                        min_y, max_y = float(
                            ys_calc.min()), float(ys_calc.max())

                        selected_y = st.slider(
                            "Select Y-coordinate to inspect (calculation space):",
                            min_value=min_y, max_value=max_y, value=(
                                min_y + max_y) / 2,
                            step=params.get('slice_thickness', 0.5), format="%.2f mm"
                        )

                        slice_start = selected_y
                        slice_end = selected_y + \
                            params.get('slice_thickness',
                                        0.5)

                        slice_df = df_for_inspector[
                            (df_for_inspector['y'] >= slice_start - FLOAT_EPSILON) &
                            (df_for_inspector['y'] <
                                slice_end - FLOAT_EPSILON)
                        ]

                        if not slice_df.empty:
                            slice_x, slice_z = slice_df['x'].to_numpy(
                            ), slice_df['z'].to_numpy()

                            area, boundary_pts, orig_pts, ds_pts = calculate_slice_profile(
                                slice_x, slice_z,
                                params.get('flat_bottom', False),
                                params.get('top_down_scan', False),
                                params.get(
                                    'area_calculation_method', "Convex Hull"),
                                params.get('alpha_shape_value', 0.02),
                                alphashape_slice_voxel_param=params.get(
                                    'alphashape_slice_voxel', 0.5)
                            )

                            fig_slice = go.Figure()
                            fig_slice.add_trace(go.Scatter(
                                x=slice_x, y=slice_z, mode='markers',
                                marker=dict(
                                    size=3, color='darkblue', opacity=0.7),
                                name=f'Slice Pts ({orig_pts})'
                            ))
                            if boundary_pts is not None:
                                fig_slice.add_trace(go.Scatter(
                                    x=boundary_pts[:, 0], y=boundary_pts[:,
                                                                            1], mode='lines',
                                    line=dict(color='red', width=2), name='Calculated Boundary'
                                ))

                            fig_slice.update_layout(
                                title=f"XZ Cross-Section @ Calc Y ≈ {selected_y:.1f} mm (Est. Area: {area:.1f} mm²)",
                                xaxis_title="Width (X, mm)", yaxis_title="Height (Z, mm)",
                                width=500, height=500, xaxis_scaleanchor="y",
                                legend=dict(orientation="h", yanchor="bottom",
                                            y=1.02, xanchor="center", x=0.5)
                            )
                            st.plotly_chart(fig_slice)
                        else:
                            st.info(
                                "No points found in the selected slice interval.")
                    else:
                        st.warning("No slice data available to inspect.")
                else:
                    st.info(
                        "Load a payload with calculation results to use the inspector.")

    with tab_details:
        st.header("Detailed Run Information")
        st.subheader("Input Parameters")
        st.json(params, expanded=True)
        st.subheader("Processing Log")
        st.text_area("Log", "".join(
            [f"{msg}\n" for msg in pipeline_log]), height=600, disabled=True, key="log_details_view")
        st.subheader("Scanner Resolution Analysis")
        resolution_data = payload.get("resolution_analysis")
        if resolution_data and 'error' not in resolution_data:
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Mean Y Spacing (Resolution)", f"{resolution_data.get('mean_spacing_mm', 0):.4f} mm")
            res_col2.metric("Unique Scan Lines Found", f"{resolution_data.get('unique_y_layers', 0):,}")
            
            with st.expander("Show Full Resolution Statistics"):
                st.json(resolution_data)
        elif resolution_data:
            st.warning(f"Resolution analysis could not be completed: {resolution_data['error']}")
        else:
            st.info("No resolution analysis data was found in this payload.")
        
if st.session_state.auto_load_enabled:
    if os.path.exists(DEFAULT_PAYLOAD_FILE):
        try:
            current_mod_time = os.path.getmtime(DEFAULT_PAYLOAD_FILE)
            # Create a unique ID for the current state of the file on disk
            current_file_id = f"{DEFAULT_PAYLOAD_FILE}-{current_mod_time}"
            
            # If the file on disk is different from the one we have loaded, reload it
            if st.session_state.last_loaded_file_id != current_file_id:
                with open(DEFAULT_PAYLOAD_FILE, "rb") as f:
                    load_payload_from_source(f, current_file_id)
        except Exception as e:
            # This can happen if the file is being written at the exact moment we check
            print(f"Auto-load check failed, will retry. Error: {e}")
    
    # Schedule the next check
    time.sleep(AUTO_REFRESH_INTERVAL_SECONDS)
    st.rerun()
