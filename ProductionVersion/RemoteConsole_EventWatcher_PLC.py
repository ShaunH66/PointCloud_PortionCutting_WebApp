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
INPUT_XYZ_FILE = "scanner_data.xyz"  # Path to the input .xyz file

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
DEFAULT_PIPELINE_PARAMS = {

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

# 5. Listener Configuration
WATCHER_INTERVAL_SECONDS = 5

# 6. Watcher Directory Configuration
XYZ_INPUT_FOLDER = "xyz_input"          # Folder to watch for new .xyz files
XYZ_PROCESSED_FOLDER = "xyz_processed"  # Folder to move files to after processing

# 7. PLC Configuration
PLC_MODE = False                  # SET TO True TO READ PARAMS FROM PLC
PLC_IP_ADDRESS = "192.168.1.10"  # IMPORTANT: Change to your PLC's IP address
PLC_PROCESSOR_SLOT = 0           # Slot of the CompactLogix/ControlLogix processor
PLC_PORTION_ARRAY_SIZE = 100      # PLC max array size for portions

# This dictionary maps your script's parameter names to the PLC tag names.
# IMPORTANT: You MUST update the PLC tag names on the right side.
PLC_TAG_MAPPING = {
    # Parameter Name in this Script : "PLC_Tag_Name"
    "target_weight": "HMI_Target_Weight",     # e.g., a REAL tag
    "total_weight": "HMI_Total_Weight",       # e.g., a REAL tag
    "start_trim": "HMI_Start_Trim",           # e.g., a REAL tag
    "end_trim": "HMI_End_Trim",             # e.g., a REAL tag
    "weight_tolerance": "HMI_Weight_Tolerance", # e.g., a REAL tag
    "no_interp": "HMI_No_Interpolation_On", # e.g., a BOOL tag (0 or 1)
    "pca_align": "HMI_PCA_Align_On",      # e.g., a BOOL tag (0 or 1)
    "blade_thickness": "HMI_Blade_Thickness", # e.g., a REAL tag (0.0 to disable)
    "slice_thickness": "HMI_Slice_Thickness", # e.g., a REAL tag

    # 0 = "Calculate from Total Weight & Volume"
    # 1 = "Input Directly"
    "density_source_selector": "HMI_Density_Source_Selector", # PLC Tag Type: DINT

    # This is the actual density value if the source is "Input Directly"
    "direct_density_g_cm3": "HMI_Direct_Density_Value",     # PLC Tag Type: REAL

    # This will be an integer from the PLC
    # 0 = "Convex Hull"
    # 1 = "Alpha Shape"
    "area_method_selector": "HMI_Area_Method_Selector"      # PLC Tag Type: DINT
}

PLC_WRITE_TAG_MAPPING = {
    "total_portions_calculated": "PC_Total_Portions_Calculated",
    "average_portion_weight": "PC_Avg_Portion_Weight",
    "first_portion_weight": "PC_First_Portion_Weight",
    "calculated_density_g_cm3": "PC_Calculated_Density",
    "total_loaf_length": "PC_Total_Loaf_Length", 
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
        

def get_params_from_plc(ip_address, slot, tag_map, log_func):
    """Connects to a Rockwell PLC and reads parameter values based on a tag map."""
    if PLC is None:
        log_func("    ...PLC_MODE is on, but pylogix library is not installed. Using defaults.")
        return {}

    log_func(f"    ...Connecting to PLC at {ip_address}...")
    plc_params = {}
    try:
        with PLC() as comm:
            comm.IPAddress = ip_address
            comm.ProcessorSlot = slot
            
            # Create a list of PLC tag names to read
            tags_to_read = list(tag_map.values())
            
            # Read all tags in one request for efficiency
            results = comm.Read(tags_to_read)
            
            if comm.StatusCode != 0:
                log_func(f"    ...ERROR communicating with PLC: {comm.Status}")
                return {}

            log_func("    ...Successfully connected and read tags.")
            
            # Map the results back to our script's parameter names
            for script_param, plc_tag in tag_map.items():
                # Find the corresponding result object
                tag_result = next((r for r in results if r.TagName == plc_tag), None)
                if tag_result and tag_result.Status == 'Success':
                    # Convert BOOLs from 0/1 to True/False
                    if isinstance(tag_result.Value, int) and script_param in ['no_interp', 'pca_align']:
                        plc_params[script_param] = bool(tag_result.Value)
                    else:
                        plc_params[script_param] = tag_result.Value
                    log_func(f"        - Read '{plc_tag}': {tag_result.Value}")
                else:
                    log_func(f"        - WARNING: Failed to read tag '{plc_tag}'. Status: {tag_result.Status if tag_result else 'Not Found'}")
    except Exception as e:
        log_func(f"    ...CRITICAL PLC_MODE ERROR: {e}")
    
    return plc_params


def write_results_to_plc(ip_address, slot, single_results_dict, portion_list_of_dicts, log_func):
    """Connects to a Rockwell PLC and writes single values and an array of UDTs."""
    if PLC is None:
        log_func("    ...PLC Write-Back skipped: pylogix library not installed.")
        return False

    log_func(f"    ...Connecting to PLC at {ip_address} to write results...")
    
    tags_to_write = []
    
    # 1. Prepare single value tags
    for result_key, plc_tag in PLC_WRITE_TAG_MAPPING.items():
        if result_key in single_results_dict:
            tags_to_write.append((plc_tag, single_results_dict[result_key]))
        else:
            log_func(f"        - WARNING: Single result key '{result_key}' not found.")

    # 2. Prepare the UDT array write
    if portion_list_of_dicts:
        for i, portion_dict in enumerate(portion_list_of_dicts):
            if i >= PLC_PORTION_ARRAY_SIZE:
                log_func(f"        - WARNING: More portions calculated ({len(portion_list_of_dicts)}) than PLC array size ({PLC_PORTION_ARRAY_SIZE}). Truncating.")
                break
            
            # For each portion, create writes for each member of the UDT
            tags_to_write.append((f"PC_Portion_Results[{i}].Start_Y", portion_dict["Start Y (mm)"]))
            tags_to_write.append((f"PC_Portion_Results[{i}].End_Y", portion_dict["End Y (mm)"]))
            tags_to_write.append((f"PC_Portion_Results[{i}].Length", portion_dict["Length (mm)"]))
            tags_to_write.append((f"PC_Portion_Results[{i}].Weight", portion_dict["Weight (g)"]))

    if not tags_to_write:
        log_func("    ...PLC Write-Back skipped: No data prepared to write.")
        return False

    try:
        with PLC() as comm:
            comm.IPAddress = ip_address
            comm.ProcessorSlot = slot
            
            response = comm.Write(tags_to_write)
            
            if comm.StatusCode != 0:
                log_func(f"    ...ERROR communicating with PLC during write: {comm.Status}")
                return False

            log_func("    ...Successfully wrote results to PLC.")
            # Optional: Log a few key values to confirm
            for tag, value in tags_to_write[:5]: # Log first 5 writes
                 log_func(f"        - Wrote to '{tag}': {value:.2f}")
            if len(tags_to_write) > 5:
                log_func("        - ... and other tags.")

        return True
    except Exception as e:
        log_func(f"    ...CRITICAL PLC Write-Back ERROR: {e}")
        return False
    
    
def process_single_file(xyz_file_path, log_messages):
    
    try: 
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
        
        # --- Step 0: Get Parameters ---
        log("\n[0/7] Acquiring processing parameters...")
        
        # Start with the script's defaults
        current_pipeline_params = DEFAULT_PIPELINE_PARAMS.copy()
        
        if PLC_MODE:
            log("    ...PLC_MODE is enabled.")
            plc_values = get_params_from_plc(PLC_IP_ADDRESS, PLC_PROCESSOR_SLOT, PLC_TAG_MAPPING, log)
            if plc_values:
                
                # Translate Density Source
                density_selector = plc_values.pop('density_source_selector', 0) # Use .pop to remove selector
                if density_selector == 1:
                    plc_values['density_source'] = "Input Directly"
                else:
                    plc_values['density_source'] = "Calculate from Total Weight & Volume"
                log(f"        - Translated Density Source Selector ({density_selector}) to: {plc_values['density_source']}")

                # Translate Area Calculation Method
                area_selector = plc_values.pop('area_method_selector', 0) # Use .pop to remove selector
                if area_selector == 1:
                    plc_values['area_calculation_method'] = "Alpha Shape"
                else:
                    plc_values['area_calculation_method'] = "Convex Hull"
                log(f"        - Translated Area Method Selector ({area_selector}) to: {plc_values['area_calculation_method']}")
                
                # Merge all other PLC values, overwriting defaults
                current_pipeline_params.update(plc_values)
                log("    ...Successfully merged parameters from PLC.")
            else:
                log("    ...Failed to get parameters from PLC, continuing with script defaults.")
        else:
            log("    ...PLC_MODE is disabled. Using script defaults.")
            
        # --- Step 1: Load Point Cloud ---
        points_numpy_array = load_xyz_to_numpy(xyz_file_path)
        if points_numpy_array is None:
            log("Aborting due to file loading error.")
            return
        current_points_df = pd.DataFrame(
            points_numpy_array, columns=['x', 'y', 'z'])
        log(f"    ...Loaded {len(current_points_df)} raw points.")
        
        processed_df = current_points_df.copy()

        # --- Step 2: PCA Alignment (Optional) ---
        if current_pipeline_params.get("pca_align"):
            log("\n[2/6] Applying PCA Alignment...")
            processed_df = align_point_cloud_with_pca(processed_df)
            log("    ...PCA Alignment Complete.")
        else:
            log("\n[2/6] PCA Alignment skipped (disabled in config).")

        # --- Step 3: Auto Downsample (Optional) ---
        if current_pipeline_params.get("enable_auto_downsample") and len(processed_df) > current_pipeline_params.get("auto_downsample_threshold", 9e9):
            threshold = current_pipeline_params.get("auto_downsample_threshold")
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
        if current_pipeline_params.get("apply_ror"):
            log("\n[4/6] Applying Radius Outlier Removal Filter...")
            if _open3d_installed:
                pcd_ror = o3d.geometry.PointCloud()
                pcd_ror.points = o3d.utility.Vector3dVector(
                    processed_df.to_numpy())

                ror_params = current_pipeline_params.copy()
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
        if current_pipeline_params.get("enable_y_normalization"):
            if not df_for_calc.empty:
                y_offset = df_for_calc['y'].min()
                df_for_calc['y'] -= y_offset
                log(
                    f"    ...Y-coords normalized for calculation. Scanner Offset: {y_offset:.2f} mm")
        else:
            log("\n[5/6] Normalization skipped (disabled in config).")

        calculation_args = {
            "total_w": current_pipeline_params.get("total_weight"), "target_w": current_pipeline_params.get("target_weight"),
            "slice_inc": current_pipeline_params.get("slice_thickness"), "no_interp": current_pipeline_params.get("no_interp"),
            "flat_bottom": current_pipeline_params.get("flat_bottom"), "top_down": current_pipeline_params.get("top_down_scan"),
            "blade_thick": current_pipeline_params.get("blade_thickness"), "w_tol": current_pipeline_params.get("weight_tolerance"),
            "start_trim": current_pipeline_params.get("start_trim"), "end_trim": current_pipeline_params.get("end_trim"),
            "area_calc_method": current_pipeline_params.get("area_calculation_method"),
            "alpha_shape_param": current_pipeline_params.get("alpha_shape_value"),
            "alphashape_slice_voxel_param": current_pipeline_params.get("alphashape_slice_voxel"),
        }
        if current_pipeline_params.get("density_source") == "Input Directly":
            calculation_args["direct_density_g_mm3"] = current_pipeline_params.get(
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
            "original_point_cloud_df": points_numpy_array,
            "processed_point_cloud_for_display_df": final_display_cloud,
            "df_for_slice_inspector": df_for_calc,
            "calculation_results": calc_results,
            "input_params_summary_for_ui": current_pipeline_params,
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
            if current_pipeline_params.get("enable_y_normalization"):
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

        # --- Step 7: Write Results back to PLC ---
        if PLC_MODE and calc_results:
            log("\n[7/7] Writing results back to PLC...")
            
            original_portions = calc_results.get("portions", [])
            scanner_offset = calc_results.get("y_offset_for_plot", 0.0)

            # --- A. Prepare single value results ---
            total_good_portions = len(original_portions) - 1 if len(original_portions) > 1 else 0
            first_portion_weight = original_portions[0]['weight'] if original_portions else 0.0
            total_weight_of_good_portions = sum(p['weight'] for p in original_portions[1:])
            avg_weight = total_weight_of_good_portions / total_good_portions if total_good_portions > 0 else 0.0
            density_g_cm3 = calc_results.get("density", 0.0) * 1000.0
            
            # Calculate total loaf length from the normalized display data
            total_loaf_length = 0.0
            if len(original_portions) > 0:
                start_y = original_portions[0]['display_start_y']
                end_y = original_portions[-1]['display_end_y']
                total_loaf_length = end_y - start_y
                
            single_results_to_write = {
                "total_portions_calculated": total_good_portions,
                "average_portion_weight": avg_weight,
                "first_portion_weight": first_portion_weight,
                "calculated_density_g_cm3": density_g_cm3,
                "total_loaf_length": total_loaf_length,
            }
            
            # --- B. Prepare the list of portion dictionaries for the UDT array ---
            # The PLC needs the "start-at-zero" normalized values
            block_start_y_real = 0.0
            if len(original_portions) > 0:
                block_start_y_real = original_portions[0]['display_start_y'] + scanner_offset

            portions_to_write = []
            for p in original_portions:
                real_start_y = p['display_start_y'] + scanner_offset
                real_end_y = p['display_end_y'] + scanner_offset
                portions_to_write.append({
                    "Portion #": p['portion_num'], # Not written, just for reference
                    "Start Y (mm)": real_start_y - block_start_y_real,
                    "End Y (mm)": real_end_y - block_start_y_real,
                    "Length (mm)": p['length'],
                    "Weight (g)": p['weight']
                })

            # --- C. Call the write function ---
            write_results_to_plc(
                PLC_IP_ADDRESS, 
                PLC_PROCESSOR_SLOT, 
                single_results_to_write, 
                portions_to_write, 
                log
            )
        else:
            log("\n[7/7] PLC Write-Back skipped (PLC_MODE is disabled or no results).")
        
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

        return True, log_messages
    
    except Exception as e:
        log(f"\n--- CRITICAL ERROR in 'process_single_file': {e} ---")
        log(traceback.format_exc())
        return False, log_messages


def start_watcher_service():
    """Main function to run the listener service."""
    print("--- Watcher Service Started ---")
    print(f"Watching for .xyz files in: ./{XYZ_INPUT_FOLDER}/")
    print("Press Ctrl+C to stop the service.")

    # Ensure necessary folders exist
    os.makedirs(XYZ_INPUT_FOLDER, exist_ok=True)
    os.makedirs(XYZ_PROCESSED_FOLDER, exist_ok=True)
    
    # Perform initial cleanup
    manage_archives(ARCHIVE_BASE_FOLDER, ARCHIVE_CLEANUP_DAYS)

    try:
        while True:
            # Find the first .xyz file in the input directory
            found_file = None
            for filename in sorted(os.listdir(XYZ_INPUT_FOLDER)): # sorted ensures FIFO processing
                if filename.lower().endswith(".xyz"):
                    found_file = filename
                    break
            
            if found_file:
                file_path = os.path.join(XYZ_INPUT_FOLDER, found_file)
                
                # Process the file
                log_capture = []
                success = process_single_file(file_path, log_capture)
                
                # Move the file after processing
                if success:
                    destination_path = os.path.join(XYZ_PROCESSED_FOLDER, found_file)
                    print(f"--- Moving processed file to: {destination_path} ---")
                    shutil.move(file_path, destination_path)
                else:
                    # Optional: Move failed files to an 'error' folder instead of 'processed'
                    error_folder = os.path.join(XYZ_PROCESSED_FOLDER, "error")
                    os.makedirs(error_folder, exist_ok=True)
                    destination_path = os.path.join(error_folder, found_file)
                    print(f"--- Moving FAILED file to: {destination_path} ---")
                    shutil.move(file_path, destination_path)

            else:
                # If no file is found, wait before scanning again
                sys.stdout.write(f"\rNo new files found. Waiting... ({time.strftime('%H:%M:%S')})")
                sys.stdout.flush()
                time.sleep(WATCHER_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n--- Watcher Service Stopped by User ---")
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---")

if __name__ == "__main__":
    start_watcher_service()
