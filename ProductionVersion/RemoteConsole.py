import numpy as np
import pandas as pd
import time
import pickle
import os
import sys
import shutil
import datetime

try:
    from FunctionLib import (
        align_point_cloud_with_pca,
        estimate_ror_radius_util_o3d,
        apply_ror_filter_to_df,
        perform_portion_calculation,
        start_o3d_visualization,
        launch_o3d_viewer_with_cuts,
        _open3d_installed,
    )

    if _open3d_installed:
        import open3d as o3d
except ImportError as e:
    print(f"FATAL ERROR: Could not import from library file: {e}")
    print("Please ensure 'Function.py' is in the same directory.")
    sys.exit(1)

# =============================================================================
# --- CONFIGURATION SECTION ---
# =============================================================================

# 1. Input Data File
INPUT_XYZ_FILE = "4-Sensors_cheese02_1000hz_01.xyz"

# 2. Output File for the Streamlit Viewer
LATEST_PAYLOAD_FILE = "latest_run_payload.pkl"

# 3. Viewer Options
# Set to True to automatically open the 3D windows after processing.
AUTO_OPEN_O3D_FLYAROUND = False
AUTO_OPEN_O3D_WITH_CUTS = False

# 4. Archiving & Cleanup Configuration 
ENABLE_ARCHIVING = True                  # Set to True to save each run in a new folder
ARCHIVE_BASE_FOLDER = "run_archives"     # Name of the main folder to store runs
ARCHIVE_CLEANUP_DAYS = 7                 # Deletes archive folders older than this many days (0 to disable cleanup)

# 5. Full Processing Pipeline Parameters
PIPELINE_PARAMS = {

    # --- Pre-processing Steps ---
    "pca_align": True,                     # Enable PCA alignment of the point cloud
    "enable_auto_downsample": True,        # Enable automatic downsampling based on point count
    "auto_downsample_threshold": 350000,   # Point count threshold for downsampling                                                   
    "voxel_size": 0.0,                     # Voxel size for downsampling (0.0 to disable)
    "enable_y_normalization": True,        # Normalize Y-coordinates for calculations

    # --- ROR Filter ---
    "apply_ror": True,                     # Apply Radius Outlier Removal
    "ror_auto_estimate_radius": True,      # Automatically find the best radius
    "ror_nb_points": 10,                   # Number of points required in the radius
    "ror_k_for_radius_est": 20,            # 'k' for k-NN distance estimation
    "ror_radius_multiplier_est": 2.0,      # Multiplier for the estimated radius
    "ror_radius": 5.0,                     # Fallback radius if estimation is off or fails

    # --- Calculation Parameters ---
    "total_weight": 3333.3,                # Total weight of the loaf in grams
    "target_weight": 500.0,                # Target weight for each portion
    "slice_thickness": 0.5,                # Thickness of each slice in mm
    "no_interp": True,                     # Disable interpolation for slice calculations
    "flat_bottom": False,                  # Use flat bottom mode
    "top_down_scan": False,                # Use top-down scan mode
    "blade_thickness": 0.0,                # Thickness of the blade in mm (0.0 to disable)    
    "weight_tolerance": 0.0,               # Weight tolerance for portion calculations (0.0 to disable)
    "start_trim": 0.0,                     # Trim from the start of the loaf
    "end_trim": 0.0,                       # Trim from the end of the loaf

    # --- Area & Density ---
    "density_source": "Calculate from Total Weight & Volume",  # Options: "Calculate from Total Weight & Volume", "Input Directly"
    "direct_density_g_cm3": 1.07,                              # Density in g/cm³ if using direct input
    "area_calculation_method": "Convex Hull",                  # Options: "Convex Hull", "Alpha Shape"
    "alpha_shape_value": 0.02,                                 # Alpha value for alpha shape calculation   
    "alphashape_slice_voxel": 5.0,                             # Voxel size for alphashape slice calculation
}

# =============================================================================
# --- SCRIPT LOGIC ---
# =============================================================================


def load_xyz_to_numpy(file_path):
    """Loads a .xyz file into a NumPy array, handling errors."""
    if not os.path.exists(file_path):
        print(f"\nERROR: File not found at '{file_path}'")
        return None
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None,
                         usecols=[0, 1, 2], names=['x', 'y', 'z'])
        df = df.dropna().astype(float)
        return df[['x', 'y', 'z']].to_numpy()
    except Exception as e:
        print(f"\nERROR: Could not load XYZ file '{file_path}': {e}")
        return None
    

def manage_archives(base_folder, days_to_keep):
    """Deletes files in the base_folder older than a specified number of days by parsing the filename."""
    if days_to_keep <= 0:
        print("    ...Archive cleanup is disabled (days_to_keep <= 0).")
        return

    print(f"\n[Cleanup] Checking for archives older than {days_to_keep} days in '{base_folder}'...")
    if not os.path.isdir(base_folder):
        print("    ...Archive folder does not exist. Nothing to clean.")
        return

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
    files_deleted = 0
    
    for filename in os.listdir(base_folder):
        file_path = os.path.join(base_folder, filename)
        if os.path.isfile(file_path) and filename.startswith("payload_") and filename.endswith(".pkl"):
            try:
                # Extract the timestamp string from the filename
                # e.g., from "payload_2024-06-22_20-15-09.pkl" -> "2024-06-22_20-15-09"
                timestamp_str = filename.replace("payload_", "").replace(".pkl", "")
                
                # Convert the string to a datetime object
                file_date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                
                # Compare the file's date (from its name) to the cutoff date
                if file_date < cutoff_date:
                    print(f"    ...Deleting old archive file: {filename} (Date: {file_date.date()})")
                    os.remove(file_path)
                    files_deleted += 1
            except ValueError:
                # This will happen if the filename is not in the expected format
                print(f"    ...Could not parse date from filename, skipping: '{filename}'")
            except Exception as e:
                print(f"    ...Could not process or delete file '{filename}': {e}")
    
    if files_deleted > 0:
        print(f"    ...Cleanup complete. Deleted {files_deleted} old archive file(s).")
    else:
        print("    ...No old archives found to delete based on filename timestamps.")
        

def run_headless_pipeline():
    print("--- Headless Pipeline Starting ---")
    start_time = time.time()

    # Use a single list to capture all log messages
    log_messages = []

    def log(message, end_line=True):
        # Print to console and append to the list
        if end_line:
            print(message)
        else:
            print(message, end="", flush=True)
        log_messages.append(message)

    manage_archives(ARCHIVE_BASE_FOLDER, ARCHIVE_CLEANUP_DAYS)
    
    # --- Step 1: Load Raw Point Cloud ---
    log(f"\n[1/6] Loading raw data from: {INPUT_XYZ_FILE}")
    points_numpy_array = load_xyz_to_numpy(INPUT_XYZ_FILE)
    if points_numpy_array is None:
        log("Aborting due to file loading error.")
        return
    current_points_df = pd.DataFrame(
        points_numpy_array, columns=['x', 'y', 'z'])
    log(f"    ...Loaded {len(current_points_df)} raw points.")

    processed_df = current_points_df.copy()

    # --- Step 2: PCA Alignment (Optional) ---
    if PIPELINE_PARAMS.get("pca_align"):
        log("\n[2/6] Applying PCA Alignment...")
        processed_df = align_point_cloud_with_pca(processed_df)
        log("    ...PCA Alignment Complete.")
    else:
        log("\n[2/6] PCA Alignment skipped (disabled in config).")

    # --- Step 3: Auto Downsample (Optional) ---
    if PIPELINE_PARAMS.get("enable_auto_downsample") and len(processed_df) > PIPELINE_PARAMS.get("auto_downsample_threshold", 9e9):
        threshold = PIPELINE_PARAMS.get("auto_downsample_threshold")
        log(f"\n[3/6] Applying Auto-Downsample (Threshold: {threshold:,})...")
        if _open3d_installed:
            pcd_ds = o3d.geometry.PointCloud()
            pcd_ds.points = o3d.utility.Vector3dVector(processed_df.to_numpy())
            ratio = threshold / len(processed_df)
            processed_df = pd.DataFrame(np.asarray(
                pcd_ds.random_down_sample(ratio).points), columns=['x', 'y', 'z'])
            log(f"    ...Downsampled to {len(processed_df)} points.")
        else:
            log("    ...Auto-Downsample skipped: Open3D not installed.")
    else:
        log("\n[3/6] Auto-Downsample skipped (not needed or disabled).")

    # --- Step 4: Radius Outlier Removal (Optional) ---
    if PIPELINE_PARAMS.get("apply_ror"):
        log("\n[4/6] Applying Radius Outlier Removal Filter...")
        if _open3d_installed:
            pcd_ror = o3d.geometry.PointCloud()
            pcd_ror.points = o3d.utility.Vector3dVector(
                processed_df.to_numpy())

            ror_params = PIPELINE_PARAMS.copy()
            if ror_params.get("ror_auto_estimate_radius"):
                est_radius, est_msg = estimate_ror_radius_util_o3d(
                    pcd_ror, ror_params.get("ror_k_for_radius_est", 20), ror_params.get(
                        "ror_radius_multiplier_est", 2.0)
                )
                log(f"    ...ROR Estimation: {est_msg}")
                if est_radius:
                    ror_params["ror_radius"] = est_radius

            n_before = len(processed_df)
            processed_df, _ = apply_ror_filter_to_df(
                processed_df, ror_params.get(
                    "ror_nb_points", 10), ror_params.get("ror_radius", 0.1)
            )
            log(f"    ...ROR Filter removed {n_before - len(processed_df)} points.")
        else:
            log("    ...ROR Filter skipped: Open3D not installed.")
    else:
        log("\n[4/6] ROR Filter skipped (disabled in config).")

    final_display_cloud = processed_df.copy()

    # --- Step 5: Y-Normalization & Final Calculation ---
    log("\n[5/6] Normalizing and Calculating Portions...")
    df_for_calc = processed_df.copy()
    y_offset = 0.0
    if PIPELINE_PARAMS.get("enable_y_normalization"):
        if not df_for_calc.empty:
            y_offset = df_for_calc['y'].min()
            df_for_calc['y'] -= y_offset
            log(
                f"    ...Y-coords normalized for calculation. Scanner Offset: {y_offset:.2f} mm")
    else:
        log("\n[5/6] Normalization skipped (disabled in config).")

    calculation_args = {
        "total_w": PIPELINE_PARAMS.get("total_weight"), "target_w": PIPELINE_PARAMS.get("target_weight"),
        "slice_inc": PIPELINE_PARAMS.get("slice_thickness"), "no_interp": PIPELINE_PARAMS.get("no_interp"),
        "flat_bottom": PIPELINE_PARAMS.get("flat_bottom"), "top_down": PIPELINE_PARAMS.get("top_down_scan"),
        "blade_thick": PIPELINE_PARAMS.get("blade_thickness"), "w_tol": PIPELINE_PARAMS.get("weight_tolerance"),
        "start_trim": PIPELINE_PARAMS.get("start_trim"), "end_trim": PIPELINE_PARAMS.get("end_trim"),
        "area_calc_method": PIPELINE_PARAMS.get("area_calculation_method"),
        "alpha_shape_param": PIPELINE_PARAMS.get("alpha_shape_value"),
        "alphashape_slice_voxel_param": PIPELINE_PARAMS.get("alphashape_slice_voxel"),
    }
    if PIPELINE_PARAMS.get("density_source") == "Input Directly":
        calculation_args["direct_density_g_mm3"] = PIPELINE_PARAMS.get(
            "direct_density_g_cm3", 0) / 1000.0

    calc_results = perform_portion_calculation(
        points_df=df_for_calc, verbose_log_func=log, **calculation_args)

    if calc_results:
        calc_results["y_offset_for_plot"] = y_offset
    else:
        log("    ...Portion calculation failed or returned no results.")
        return

    # --- Step 6: Save Payload & Display Results ---
    # Create the final payload dictionary
    output_payload = {
        "original_point_cloud_df": current_points_df,
        "processed_point_cloud_for_display_df": final_display_cloud,
        "df_for_slice_inspector": df_for_calc,
        "calculation_results": calc_results,
        "input_params_summary_for_ui": PIPELINE_PARAMS,
        "pipeline_log": log_messages,
    }
    
    # Create a unique filename using a timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_filename = f"payload_{timestamp}.pkl"

    # Determine the final save path
    if ENABLE_ARCHIVING:
        os.makedirs(ARCHIVE_BASE_FOLDER, exist_ok=True)
        output_path = os.path.join(ARCHIVE_BASE_FOLDER, unique_filename)
        log(f"\n[6/6] Archiving enabled. Saving payload to '{output_path}'...")
    else:
        # Save to the root directory with a unique name
        output_path = unique_filename
        log(f"\n[6/6] Archiving disabled. Saving uniquely named payload to '{output_path}'...")

    try:
        # Save the uniquely named file
        with open(output_path, "wb") as f:
            pickle.dump(output_payload, f)
        log("    ...Payload saved successfully.")
        
        # Create/overwrite the "latest" file for the viewer to find easily
        shutil.copyfile(output_path, LATEST_PAYLOAD_FILE)
        log(f"    ...Updated '{LATEST_PAYLOAD_FILE}' for viewer app.")
    except Exception as e:
        log(f"    ...ERROR saving payload file: {e}")

    # --- Print Summary to Console ---
    print("\n--- Calculation Results Summary ---")
    print(f"Status: {calc_results.get('status')}")

    original_portions = calc_results.get("portions", [])
    if original_portions:

        display_data = []
        if PIPELINE_PARAMS.get("enable_y_normalization"):
            # Apply start-at-zero logic for display
            block_start_y_real = 0.0
            if len(original_portions) > 0:
                block_start_y_real = original_portions[0]['display_start_y'] + y_offset

            for p in original_portions:
                real_start_y = p['display_start_y'] + y_offset
                real_end_y = p['display_end_y'] + y_offset
                display_data.append({
                    "Portion #": p['portion_num'],
                    "Start Y (mm)": real_start_y - block_start_y_real,
                    "End Y (mm)": real_end_y - block_start_y_real,
                    "Length (mm)": p['length'],
                    "Weight (g)": p['weight']
                })
        else:
            # If normalization is disabled, show the original scanner coordinates
            for p in original_portions:
                display_data.append({
                    "Portion #": p['portion_num'],
                    # y_offset will be 0
                    "Start Y (mm)": p['display_start_y'] + y_offset,
                    # y_offset will be 0
                    "End Y (mm)": p['display_end_y'] + y_offset,
                    "Length (mm)": p['length'],
                    "Weight (g)": p['weight']
                })

        summary_df = pd.DataFrame(display_data)
        pd.options.display.float_format = '{:,.2f}'.format
        print(summary_df.to_string(index=False))

    # --- Optional: Launch Open3D Viewers ---
    if AUTO_OPEN_O3D_FLYAROUND and _open3d_installed:
        print("\nLaunching Open3D Fly-Around Viewer...")
        start_o3d_visualization(final_display_cloud)

    if AUTO_OPEN_O3D_WITH_CUTS and _open3d_installed:
        print("\nLaunching Open3D Viewer with Cuts...")
        o3d_portions = []
        for p in original_portions:
            p_copy = p.copy()
            p_copy['display_start_y'] += y_offset
            p_copy['display_end_y'] += y_offset
            if 'cut_y' in p_copy:
                p_copy['cut_y'] += y_offset
            o3d_portions.append(p_copy)

        launch_o3d_viewer_with_cuts(
            final_display_cloud,
            o3d_portions,
            calc_results.get("calc_start_y", 0) + y_offset,
            calc_results.get("calc_end_y", 0) + y_offset,
            0
        )

    end_time = time.time()
    print(
        f"\n--- Headless Pipeline Finished in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    run_headless_pipeline()
