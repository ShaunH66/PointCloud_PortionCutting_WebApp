import numpy as np
import pandas as pd
import time
import pickle
import os
import sys
import shutil
import datetime
import traceback

# --- Import PLC and Point Cloud Libraries ---
try:
    from pylogix import PLC
except ImportError:
    print("WARNING: pylogix library not found. PLC_MODE will be unavailable.")
    PLC = None

try:
    # Import functions from the library, including the new ones for cheese wheels
    from functionlib import (
        align_point_cloud_with_pca,
        perform_wedge_calculation,
        generate_deformed_cheese_wheel,
        launch_o3d_viewer_with_wedge_cuts,
        load_point_cloud_from_file,
        estimate_ror_radius_util_o3d,
        _open3d_installed,
    )

    if _open3d_installed:
        import open3d as o3d

except ImportError as e:
    print(f"FATAL ERROR: Could not import from library file: {e}")
    print("Please ensure 'functionlib.py' is updated and in the same directory.")
    sys.exit(1)

# =============================================================================
# --- CHEESE WHEEL CONFIGURATION SECTION ---
# =============================================================================

# 1. Output File for the Streamlit Viewer
LATEST_PAYLOAD_FILE = "latest_run_payload_wheel.pkl"
GENERATE_TEST_FILE = True  # Set to True to generate a test cheese wheel file

# 2. Viewer Options
AUTO_OPEN_O3D_WITH_CUTS = True  # Set to True to see the wedge cuts

# 3. Archiving & Cleanup Configuration
ENABLE_ARCHIVING = True
ARCHIVE_BASE_FOLDER = "run_archives_wheel"
ARCHIVE_CLEANUP_DAYS = 7

# 4. Full Processing Pipeline Parameters for Cheese Wheels
DEFAULT_PIPELINE_PARAMS = {
    # --- Pre-processing Steps ---
    "pca_align": False,                    # PCA might not be ideal for wheels unless they are very off-center. Assuming the wheel is scanned relatively flat.
    "enable_auto_downsample": True,        # Enable automatic downsampling based on point count
    "auto_downsample_threshold": 350000,   # Point count threshold for downsampling                                                   
    "voxel_size": 0.0,                     # Voxel size for downsampling (0.0 to disable)
    
    # --- ROR Filter ---
    "apply_ror": False,                     # Apply Radius Outlier Removal
    "ror_auto_estimate_radius": True,      # Automatically find the best radius
    "ror_nb_points": 10,                   # Number of points required in the radius
    "ror_k_for_radius_est": 20,            # 'k' for k-NN distance estimation
    "ror_radius_multiplier_est": 2.0,      # Multiplier for the estimated radius
    "ror_radius": 5.0,                     # Fallback radius if estimation is off or fails
    "ror_samples": 500,                    # Number of samples to use for radius estimation
    
    # --- Calculation Parameters for Wedges ---
    "waste_redistribution": False,         # If True, will redistribute waste to the last wedge
    "guarantee_overweight": True,          # If True, will ensure each wedge is at least the target weight
    "total_weight": 7200.0,                # Total weight of the cheese wheel in grams
    "target_weight": 250.0,                # Target weight for each wedge in grams
    "volume_slice_thickness": 0.5,         # Thickness of virtual slices for volume calculation (mm)
    "blade_thickness_deg": 0.0,            # Thickness of the blade in degrees (e.g., 1.5 degrees)
    "start_angle_deg_offset": 0.0,         # An angle to start cutting from (e.g. to avoid a clamp). 90 degree would be Y=0.
    "num_angular_slices": 3600,            # Number of angular slices for profiling (higher is more accurate)

    # --- Area & Density ---
    "density_source": "Calculate from Total Weight & Volume",  # Options: "Calculate from Total Weight & Volume", "Input Directly"
    "direct_density_g_cm3": 1.05,                              # Density in g/cm³ if using direct input
    "area_calculation_method": "Convex Hull",                  # Options: "Convex Hull", "Alpha Shape"
    "alpha_shape_value": 0.02,
    "alphashape_slice_voxel": 5.0,
}

# 5. Listener Configuration
WATCHER_INTERVAL_SECONDS = 5
XYZ_INPUT_FOLDER = "xyz_input_wheel"
XYZ_PROCESSED_FOLDER = "xyz_processed_wheel"

# 6. PLC Configuration
PLC_MODE = False                # SET TO True TO READ PARAMS FROM PLC
PLC_IP_ADDRESS = "192.168.1.10"
PLC_PROCESSOR_SLOT = 0
PLC_PORTION_ARRAY_SIZE = 50     # PLC max array size for wedges

PLC_HEARTBEAT_TAG = "PC_Heartbeat" # PLC Tag Type: DINT
PLC_STATUS_TAG = "PC_Status"       # PLC Tag Type: DINT
STATUS_IDLE = 1
STATUS_PROCESSING = 2
STATUS_SUCCESS = 3
STATUS_ERROR = 4

# This dictionary maps your script's parameter names to the PLC tag names.
# IMPORTANT: You MUST update the PLC tag names on the right side.
PLC_TAG_MAPPING = {
    "target_weight": "HMI_Wedge_Target_Weight",                 # PLC Tag Type: REAL
    "total_weight": "HMI_Wheel_Total_Weight",                   # PLC Tag Type: REAL
    "blade_thickness_deg": "HMI_Blade_Thickness_Deg",           # PLC Tag Type: REAL
    "start_angle_deg_offset": "HMI_Start_Angle_Offset",         # PLC Tag Type: REAL
    "waste_redistribution": "HMI_Waste_Redistribution_On",      # PLC Tag Type: BOOL
    "completion_counter": "PC_Completion_Counter",              # PLC Tag Type: DINT
    "density_source_selector": "HMI_Density_Source_Selector",   # PLC Tag Type: DINT
    "direct_density_g_cm3": "HMI_Direct_Density_Value",         # PLC Tag Type: REAL
    "guarantee_overweight": "HMI_Guarantee_Overweight_On",      # PLC Tag Type: BOOL
}

# IMPORTANT: PLC Write-back mapping for CHEESE WHEEL results
PLC_WRITE_TAG_MAPPING = {
    "completion_counter": "PC_Completion_Counter",              # PLC Tag Type: DINT 
    "total_portions_calculated": "PC_Total_Wedges_Calculated",  # PLC Tag Type: DINT
    "average_portion_weight": "PC_Avg_Wedge_Weight",            # PLC Tag Type: REAL
    "calculated_density_g_cm3": "PC_Calculated_Density",        # PLC Tag Type: REAL
    "yield_percentage": "PC_Yield_Percentage",                  # PLC Tag Type: REAL
}

#  --- PLC User-Defined Type (UDT) for Rockwell Studio 5000 ---
#
#  Create a UDT with the following structure. This matches the data sent
#  by the final version of the script.
#
#  UDT Name: PC_Wedge_Data
#  _____________________________________________________________________________
#  | Member Name        | Data Type | Description                               |
#  |--------------------|-----------|-------------------------------------------|
#  | PortionNumber      | DINT      | The portion number (e.g., 1, 2, 3...).    |
#  |                    |           | A value of 0 indicates the Balance wedge. |
#  |--------------------|-----------|-------------------------------------------|
#  | StartAngle         | REAL      | The absolute starting angle of the wedge  |
#  |                    |           | (0-360°). Useful for HMI display.         |
#  |--------------------|-----------|-------------------------------------------|
#  | EndAngle           | REAL      | The absolute ending angle of the wedge    |
#  |                    |           | (0-360°). Also good for HMI display.      |
#  |--------------------|-----------|-------------------------------------------|
#  | IncrementalAngle   | REAL      | **The main command for the PLC.** This is |
#  |                    |           | the number of degrees the turntable       |
#  |                    |           | should rotate for this specific cut.      |
#  |--------------------|-----------|-------------------------------------------|
#  | CalculatedWeight   | REAL      | The calculated weight of the resulting    |
#  |                    |           | wedge in grams.                           |
#  _____________________________________________________________________________
#
#  --- PLC Array Tag ---
#
#  Create an ARRAY tag in the PLC's Controller Tags database:
#  - Name:               PC_Wedge_Results
#  - Data Type:          PC_Wedge_Data (the UDT you just created)
#  - Dimensions (Dim 0): 50 (or a value matching PLC_PORTION_ARRAY_SIZE)

# =============================================================================
# --- SCRIPT LOGIC ---
# =============================================================================


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


def update_plc_status(ip, slot, heartbeat=None, status=None):
    """Updates PLC status and heartbeat tags."""
    if not PLC_MODE or PLC is None:
        return
    tags = []
    if heartbeat is not None:
        tags.append((PLC_HEARTBEAT_TAG, heartbeat))
    if status is not None:
        tags.append((PLC_STATUS_TAG, status))
    if not tags:
        return
    try:
        with PLC() as comm:
            comm.IPAddress, comm.ProcessorSlot = ip, slot
            comm.Write(tags)
    except Exception as e:
        print(f"\nWARNING: Could not update PLC status. Error: {e}")


def get_params_from_plc(ip, slot, tag_map, log):
    """Reads parameters from the PLC."""
    if PLC is None:
        log("    ...PLC_MODE is on, but pylogix is not installed. Using defaults.")
        return {}
    log(f"    ...Connecting to PLC at {ip}...")
    params = {}
    try:
        with PLC() as comm:
            comm.IPAddress, comm.ProcessorSlot = ip, slot
            results = comm.Read(list(tag_map.values()))
            for p_name, tag in tag_map.items():
                res = next((r for r in results if r.TagName == tag), None)
                if res:
                    params[p_name] = bool(res.Value) if isinstance(
                        res.Value, int) and 'On' in tag else res.Value
                    log(f"        - Read '{tag}': {res.Value}")
                else:
                    log(f"        - WARNING: Failed to read tag '{tag}'.")
    except Exception as e:
        log(f"    ...CRITICAL PLC_MODE ERROR: {e}")
    return params


def write_results_to_plc(ip, slot, single_results, wedge_list, log):
    """
    Writes wedge results to the PLC using the NEW UDT structure.
    """
    if not PLC_MODE or PLC is None:
        log("    ...PLC Write-Back skipped: pylogix not installed or PLC_MODE is off.")
        return False
    log(f"    ...Connecting to PLC at {ip} to write WEDGE results...")

    tags_to_write = []

    # 1. Prepare single value tags (unchanged)
    for key, tag in PLC_WRITE_TAG_MAPPING.items():
        if key in single_results:
            tags_to_write.append((tag, single_results[key]))

    # 2. Prepare the UDT array write for wedges
    for i, wedge in enumerate(wedge_list):
        if i >= PLC_PORTION_ARRAY_SIZE:
            log(
                f"        - WARNING: More wedges ({len(wedge_list)}) than PLC array size ({PLC_PORTION_ARRAY_SIZE}). Truncating.")
            break

        # Handle Portion # (sending 0 for "Balance")
        portion_num = wedge["Portion #"]
        if not isinstance(portion_num, int):
            portion_num = 0  # Use 0 to signify the Balance wedge

        tags_to_write.append(
            (f"PC_Wedge_Results[{i}].PortionNumber", portion_num))
        tags_to_write.append(
            (f"PC_Wedge_Results[{i}].StartAngle", wedge["Start Angle (deg)"]))
        tags_to_write.append(
            (f"PC_Wedge_Results[{i}].EndAngle", wedge["End Angle (deg)"]))
        tags_to_write.append(
            (f"PC_Wedge_Results[{i}].IncrementalAngle", wedge["Incremental Angle (deg)"]))
        tags_to_write.append(
            (f"PC_Wedge_Results[{i}].CalculatedWeight", wedge["Weight (g)"]))

    if not tags_to_write:
        log("    ...PLC Write-Back skipped: No data prepared to write.")
        return False

    try:
        with PLC() as comm:
            comm.IPAddress, comm.ProcessorSlot = ip, slot
            comm.Write(tags_to_write)
            log("    ...Successfully wrote wedge results to PLC.")
        return True
    except Exception as e:
        log(f"    ...CRITICAL PLC Write-Back ERROR: {e}")
        return False


def process_single_file(file_path, log_messages):
    """Main processing pipeline for a single cheese wheel point cloud."""
    try:
        print("\n\n--- Cheese Wheel File Found: Pipeline Starting ---")
        start_time = time.time()

        def log(message, end_line=True):
            print(message, end='\n' if end_line else '', flush=not end_line)
            log_messages.append(message)

        # --- Step 0: Get Parameters ---
        log("\n[0/7] Acquiring processing parameters...")
        params = DEFAULT_PIPELINE_PARAMS.copy()
        if PLC_MODE:
            log("    ...PLC_MODE is enabled.")
            plc_values = get_params_from_plc(
                PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, PLC_TAG_MAPPING, log)
            if plc_values:
                # Handle selectors if they exist
                if 'density_source_selector' in plc_values:
                    params['density_source'] = "Input Directly" if plc_values.pop(
                        'density_source_selector') == 1 else "Calculate from Total Weight & Volume"
                params.update(plc_values)
                log("    ...Successfully merged parameters from PLC.")
        else:
            log("    ...PLC_MODE is disabled. Using script defaults.")

        # --- Step 1: Load Raw Point Cloud ---
        log(f"\n[1/7] Loading raw data from: {os.path.basename(file_path)}")
        points_numpy = load_point_cloud_from_file(file_path)
        if points_numpy is None or points_numpy.shape[0] == 0:
            log("Aborting: file loading error or empty cloud.")
            return False, log_messages

        points_df = pd.DataFrame(points_numpy, columns=['x', 'y', 'z'])
        log(f"    ...Loaded {len(points_df)} raw points.")

        # --- Step 2: Pre-processing (Optional PCA) ---
        if params.get("pca_align", False):
            log("\n[2/7] Applying PCA Alignment...")
            points_df = align_point_cloud_with_pca(points_df)
        else:
            log("\n[2/7] PCA Alignment skipped.")

        # --- Convert to Open3D object ONCE for next subsequent steps ---
        pcd_for_processing = o3d.geometry.PointCloud()
        pcd_for_processing.points = o3d.utility.Vector3dVector(points_df.values)
        
       # --- Step 3: Auto Downsample (Operates on the Open3D object) ---
        if params['enable_auto_downsample'] and len(pcd_for_processing.points) > params['auto_downsample_threshold']:
            threshold = params['auto_downsample_threshold']
            log(f"\n[3/7] Applying Auto-Downsample (Threshold: {threshold:,})...")
            auto_downsample_starttime = time.time()
            if _open3d_installed:
                ratio = threshold / len(pcd_for_processing.points)
                # Overwrite the pcd object with the downsampled version
                pcd_for_processing = pcd_for_processing.random_down_sample(ratio)
                log(f"    ...Downsampled to {len(pcd_for_processing.points):,} points.")
                log(f"    ...Auto-Downsample took {time.time() - auto_downsample_starttime:.2f} seconds.")
            else:
                log("    ...Auto-Downsample skipped: Open3D not installed.")
        else:
            log("\n[3/7] Auto-Downsample skipped (not needed or disabled).")

        # --- Step 4: Radius Outlier Removal (Operates on the same Open3D object) ---
        if params['apply_ror']:
            log("\n[4/7] Applying Radius Outlier Removal Filter...")
            if _open3d_installed:
                n_before = len(pcd_for_processing.points)
                ror_params = params.copy()
                if ror_params.get("ror_auto_estimate_radius"):
                    estimation_args = {
                        "k_neighbors": ror_params.get("ror_k_for_radius_est", 20),
                        "mult": ror_params.get("ror_radius_multiplier_est", 2.0),
                        "ror_samples": ror_params.get("ror_samples", 500)
                    }
                    ror_starttime = time.time()
                    # Call the function with the clean arguments
                    est_radius, est_msg = estimate_ror_radius_util_o3d(
                        pcd_for_processing, **estimation_args
                    )
                    log(f"    ...ROR Estimation: {est_msg}")
                    if est_radius: ror_params["ror_radius"] = est_radius
                
                pcd_filtered, _ = pcd_for_processing.remove_radius_outlier(
                    nb_points=int(ror_params.get("ror_nb_points", 10)),
                    radius=float(ror_params.get("ror_radius", 5.0))
                )
                log(f"    ...ROR Filter removed {n_before - len(pcd_filtered.points)} points.")
                log(f"    ...ROR Filter took {time.time() - ror_starttime:.2f} seconds.")
                pcd_for_processing = pcd_filtered 
            else:
                log("    ...ROR Filter skipped: Open3D not installed.")
        else:
            log("\n[4/7] ROR Filter skipped (disabled in config).")
        
        # --- Final Conversion back to DataFrame for calculation ---
        points_df = pd.DataFrame(np.asarray(pcd_for_processing.points), columns=['x', 'y', 'z'])
        
        # --- Step 5: Perform Wedge Calculation ---
        log("\n[5/7] Calculating Wedges...")
        wedge_calc_args = {
            "total_w": params["total_weight"],
            "target_w": params["target_weight"],
            "slice_inc_mm": params["volume_slice_thickness"],
            "blade_thick_deg": params["blade_thickness_deg"],
            "start_angle_offset_deg": params["start_angle_deg_offset"],
            "num_angular_slices": params["num_angular_slices"],
            "redistribute_waste": params["waste_redistribution"],
            "guarantee_overweight": params["guarantee_overweight"],
            "area_calc_method": params["area_calculation_method"],
            "alpha_shape_param": params["alpha_shape_value"],
        }
        if params["density_source"] == "Input Directly":
            wedge_calc_args["direct_density_g_cm3"] = params["direct_density_g_cm3"]

        calc_results = perform_wedge_calculation(
            points_df=points_df,
            verbose_log_func=log,
            **wedge_calc_args
        )
        if not calc_results or not calc_results.get("portions"):
            log("    ...Wedge calculation failed or returned no results.")
            return False, log_messages

        # --- Step 6: Write Results back to PLC ---
        if PLC_MODE:
            log("\n[6/7] Writing results back to PLC...")
            wedges = calc_results.get("portions", [])
            total_good_portions = len(wedges)
            avg_weight = sum(p['Weight (g)'] for p in wedges) / \
                total_good_portions if total_good_portions > 0 else 0
            yield_perc = (avg_weight * total_good_portions) / \
                params['total_weight'] * \
                100 if params['total_weight'] > 0 else 0

            single_results = {
                "completion_counter": (params.get("completion_counter", 0) + 1) % 2147483647,
                "total_portions_calculated": total_good_portions,
                "average_portion_weight": avg_weight,
                "calculated_density_g_cm3": calc_results.get("density_g_cm3", 0.0),
                "yield_percentage": yield_perc
            }
            write_results_to_plc(
                PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, single_results, wedges, log)
        else:
            log("\n[6/7] PLC Write-Back skipped.")

        # --- Step 5: Save Payload & Display Results ---
        log(f"\n[7/7] Finalizing and saving results...")
        output_payload = {
            "original_point_cloud_df": points_numpy,
            "processed_point_cloud_for_display_df": points_df,
            "calculation_results": calc_results,
            "input_params_summary_for_ui": params,
            "pipeline_log": log_messages,
        }
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_filename = f"payload_{timestamp}.pkl"
        output_path = os.path.join(
            ARCHIVE_BASE_FOLDER, unique_filename) if ENABLE_ARCHIVING else unique_filename
        if ENABLE_ARCHIVING:
            os.makedirs(ARCHIVE_BASE_FOLDER, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(output_payload, f)
        shutil.copyfile(output_path, LATEST_PAYLOAD_FILE)
        log(f"    ...Payload saved to '{output_path}' and updated '{LATEST_PAYLOAD_FILE}'.")

        print("\n--- Wedge Calculation Results Summary ---")
        summary_df = pd.DataFrame(calc_results.get("portions", []))
        if not summary_df.empty:
            pd.options.display.float_format = '{:,.2f}'.format
            print(summary_df.to_string(index=False))

        log(f"\n--- Pipeline completed in {time.time() - start_time:.2f} seconds ---")

        if AUTO_OPEN_O3D_WITH_CUTS and _open3d_installed:
            print("\nLaunching Open3D Viewer with Wedge Cuts...")
            launch_o3d_viewer_with_wedge_cuts(
                points_df, calc_results.get("portions", []))

        return True, log_messages

    except Exception as e:
        log_messages.append(
            f"\n--- CRITICAL ERROR in 'process_single_file': {e} ---")
        log_messages.append(traceback.format_exc())
        print(log_messages[-2])
        print(log_messages[-1])
        return False, log_messages


def start_watcher_service():
    """Main function to run the listener service for cheese wheels."""
    print("--- Cheese Wheel Watcher Service Started ---")
    supported_ext = ('.xyz', '.pcd', 'ply', '.csv')
    print(
        f"Watching for {', '.join(supported_ext)} files in: ./{XYZ_INPUT_FOLDER}/")

    os.makedirs(XYZ_INPUT_FOLDER, exist_ok=True)
    os.makedirs(XYZ_PROCESSED_FOLDER, exist_ok=True)
    
    if GENERATE_TEST_FILE:
        test_file_path = os.path.join(XYZ_INPUT_FOLDER, "deformed_wheel.xyz")
        if not os.path.exists(test_file_path):
            print("\n--- Generating a deformed cheese wheel for testing... ---")
            wheel_cloud = generate_deformed_cheese_wheel(
                num_points=400000,
                nominal_radius=150,  # Corresponds to 300mm diameter
                nominal_height=100,
                radial_waviness=2.5,  # Wobble in radius
                height_waviness=3.0,  # Wobble in height
                tilt_deg=1.0  # Leans slightly
            )
            np.savetxt(test_file_path, wheel_cloud, fmt='%.4f')
            print(
                f"--- Test file 'deformed_wheel.xyz' created in '{XYZ_INPUT_FOLDER}'. ---")

    print("Press Ctrl+C to stop the service.")
    manage_archives(ARCHIVE_BASE_FOLDER, ARCHIVE_CLEANUP_DAYS)
    heartbeat = 0
    try:
        update_plc_status(PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT,
                          heartbeat=heartbeat, status=STATUS_IDLE)
        while True:
            heartbeat = (heartbeat + 1) % 2147483647
            update_plc_status(
                PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, heartbeat=heartbeat)

            found_file = next((f for f in sorted(os.listdir(
                XYZ_INPUT_FOLDER)) if f.lower().endswith(supported_ext)), None)

            if found_file:
                update_plc_status(
                    PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, status=STATUS_PROCESSING)
                file_path = os.path.join(XYZ_INPUT_FOLDER, found_file)
                logs = []
                success, _ = process_single_file(file_path, logs)

                dest_folder = XYZ_PROCESSED_FOLDER if success else os.path.join(
                    XYZ_PROCESSED_FOLDER, "error")
                if not success:
                    os.makedirs(dest_folder, exist_ok=True)

                shutil.move(file_path, os.path.join(dest_folder, found_file))
                print(f"--- Moved processed file to: {dest_folder} ---")

                update_plc_status(PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT,
                                  status=STATUS_SUCCESS if success else STATUS_ERROR)
                update_plc_status(
                    PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, status=STATUS_IDLE)
            else:
                sys.stdout.write(
                    f"\rNo new files found. Waiting... (Heartbeat: {heartbeat}) ({time.strftime('%H:%M:%S')})")
                sys.stdout.flush()
                time.sleep(WATCHER_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n--- Watcher Service Stopped by User ---")
        update_plc_status(PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, status=0)
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---")


if __name__ == "__main__":
    start_watcher_service()
