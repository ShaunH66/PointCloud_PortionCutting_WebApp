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
# 1. PREREQUISITES:
#    - You must have the `python-snap7` library installed (`pip install python-snap7`).
#    - You MUST download the native Snap7 library binary for your system.
#      - Go to: http://snap7.sourceforge.net/
#      - Download the file that matches your OS and architecture (e.g., `snap7-1.4.2-win64.zip` for 64-bit Windows).
#      - Unzip the file and place the `snap7.dll` (for Windows) or `libsnap7.so` (for Linux)
#        in the SAME DIRECTORY as this script.

try:
    import snap7
    from snap7.util import get_real, set_real, get_dint, set_dint, get_bool, set_bool
except ImportError:
    snap7 = None
   
try:
    from functionlib import (
        align_point_cloud_with_pca,
        estimate_ror_radius_util_o3d,
        apply_ror_filter_to_df,
        perform_portion_calculation,
        start_o3d_visualization,
        launch_o3d_viewer_with_cuts,
        load_point_cloud_from_file,
        calculate_with_waste_redistribution,
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

# 1. Supported File Formats                 
# Extension	    Primary Loader	    Fallback Loader	    Notes
# .xyz	        Open3D	                Pandas	        Most common text format. Fast loading with Open3D.
# .pcd	        Open3D	                  -	            Standard PCL format. Supports binary for speed and size.
# .ply	        Open3D	                  -	            Standard format for 3D scanned data. Supports binary.
# .csv	        Pandas	                  -	            Flexible text format. Can have headers or not.
# .xlsx	        Pandas	                  -	            Requires openpyxl. Reads the first sheet.
# .xls	        Pandas	                  -	            Legacy Excel format. Reads the first sheet.

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
    "waste_redistribution": False,         # Enable waste redistribution
    "total_weight": 3333.3,                # Total weight of the loaf in grams
    "target_weight": 250.0,                # Target weight for each portion
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
PLC_MODE = False
PLC_IP_ADDRESS = "192.168.1.20"  # Siemens PLC IP
PLC_RACK = 0
PLC_SLOT = 1                      # Typically 1 for S7-1200/1500
PLC_DB_NUMBER = 10                # The single DB for all data
PLC_PORTION_ARRAY_SIZE = 100

PLC_HEARTBEAT_TAG = "PC_Heartbeat" # These are now logical keys
PLC_STATUS_TAG = "PC_Status"
STATUS_IDLE = 1
STATUS_PROCESSING = 2
STATUS_SUCCESS = 3
STATUS_ERROR = 4

# Siemens Tag Mapping (Byte Offsets within the DB)
# IMPORTANT: Update the `SIEMENS_TAG_MAPPING` dictionary. This defines the memory layout of the Data Block (DB)
#      that this script will read from and write to. The byte offsets and data types must EXACTLY match
#      the structure of the DB in your TIA Portal or Step 7 project.

SIEMENS_TAG_MAPPING = {
    "target_weight": [0, "REAL"], 
    "total_weight": [4, "REAL"],
    "start_trim": [8, "REAL"], 
    "end_trim": [12, "REAL"],
    "no_interp": [16, "BOOL", 0], 
    "pca_align": [16, "BOOL", 1],
    "waste_redistribution": [16, "BOOL", 2], 
    "blade_thickness": [20, "REAL"],
    "slice_thickness": [24, "REAL"], 
    "density_source_selector": [28, "DINT"],
    "direct_density_g_cm3": [32, "REAL"], 
    "area_method_selector": [36, "DINT"],
    
    # PC Status & Results Area
    "completion_counter": [50, "DINT"],
    "plc_heartbeat_tag": [54, "DINT"], 
    "plc_status_tag": [58, "DINT"],   
    "total_portions_calculated": [62, "DINT"],
    "average_portion_weight": [66, "REAL"], 
    "first_portion_weight": [70, "REAL"],
    "calculated_density_g_cm3": [74, "REAL"], 
    "total_loaf_length": [78, "REAL"],
    "yield_percentage": [82, "REAL"],

    # Portion Results Array (UDT)
    "portion_array_start_offset": 100,
}

# This maps the internal structure of your "Portion_Data" UDT in the Siemens PLC
SIEMENS_PORTION_UDT_MAP = {
    "Start Y (mm)": [0, "REAL"], "End Y (mm)": [4, "REAL"],
    "Length (mm)": [8, "REAL"], "Weight (g)": [12, "REAL"],
}
SIEMENS_UDT_SIZE_BYTES = 16

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
        
        
# In run_pipeline_siemens.py

def update_plc_status(ip, rack, slot, db_num, tag_map, heartbeat_val=None, status_val=None):
    """Performs a targeted read-modify-write for just the status/heartbeat tags in a Siemens DB."""
    if not PLC_MODE or snap7 is None:
        return
    
    tags_to_update = {}
    if heartbeat_val is not None:
        tags_to_update[PLC_HEARTBEAT_TAG] = heartbeat_val
    if status_val is not None:
        tags_to_update[PLC_STATUS_TAG] = status_val
    if not tags_to_update:
        return

    client = snap7.client.Client()
    try:
        client.connect(ip, rack, slot)
        if not client.get_connected():
            print(f"\nWARNING: Could not connect to Siemens PLC for status update.")
            return

        # Determine the smallest memory area to read/write for efficiency
        min_offset = float('inf')
        max_offset = float('-inf')
        for key in tags_to_update:
            if key in tag_map:
                offset, dtype, *_ = tag_map[key]
                min_offset = min(min_offset, offset)
                size = 4 if dtype == "DINT" else 1 # Assuming status/heartbeat are DINT
                max_offset = max(max_offset, offset + size)
        
        if min_offset > max_offset: return

        size_to_read = max_offset - min_offset
        db_data = client.db_read(db_num, min_offset, size_to_read)

        # Modify the bytearray in place
        for key, value in tags_to_update.items():
            if key in tag_map:
                offset, dtype, *_ = tag_map[key]
                relative_offset = offset - min_offset
                if dtype == "DINT": set_dint(db_data, relative_offset, int(value))

        client.db_write(db_num, min_offset, db_data)
    except Exception as e:
        print(f"\nWARNING: Could not update Siemens PLC status. Error: {e}")
    finally:
        if client.get_connected(): client.disconnect()


def get_params_from_plc(ip, rack, slot, db_num, tag_map, log_func):
    """Connects to a Siemens S7 PLC and reads parameter values from a DB."""
    if snap7 is None:
        log_func("    ...Siemens mode is on, but python-snap7 is not installed.")
        return {}

    log_func(f"    ...Connecting to Siemens PLC at {ip}...")
    plc_params = {}
    client = snap7.client.Client()
    try:
        client.connect(ip, rack, slot)
        if not client.get_connected():
            log_func(f"    ...ERROR: Could not connect to Siemens PLC at {ip}.")
            return {}

        max_offset = max(offset + (4 if dtype in ["REAL", "DINT"] else 1) for offset, dtype, *_ in tag_map.values())
        db_data = client.db_read(db_num, 0, max_offset)
        log_func("    ...Successfully connected and read DB.")

        for script_param, (offset, dtype, *bit) in tag_map.items():
            value = None
            try:
                if dtype == "REAL": value = get_real(db_data, offset)
                elif dtype == "DINT": value = get_dint(db_data, offset)
                elif dtype == "BOOL": value = get_bool(db_data, offset, bit[0])
                
                if value is not None:
                    plc_params[script_param] = value
                    log_func(f"        - Read {script_param} (DB{db_num}.{offset}): {value}")
            except Exception as e:
                log_func(f"        - WARNING: Failed to parse {script_param} from DB data. Error: {e}")
                
    except Exception as e:
        log_func(f"    ...CRITICAL Siemens PLC_MODE ERROR: {e}")
    finally:
        if client.get_connected(): client.disconnect()
    
    return plc_params


def write_results_to_plc(ip, rack, slot, db_num, single_results_dict, portion_list_of_dicts, log_func):
    """Connects to a Siemens S7 PLC and writes all results back to a DB."""
    if snap7 is None:
        log_func("    ...Siemens Write-Back skipped: python-snap7 not installed.")
        return False

    log_func(f"    ...Connecting to Siemens PLC at {ip} to write results...")
    
    client = snap7.client.Client()
    try:
        client.connect(ip, rack, slot)
        if not client.get_connected():
            log_func(f"    ...ERROR: Could not connect to Siemens PLC for writing.")
            return False

        # Calculate the total size of the DB area we need to modify
        max_single_offset = max(offset + 4 for key, (offset, _, *_) in SIEMENS_TAG_MAPPING.items() if key in single_results_dict)
        portion_array_start = SIEMENS_TAG_MAPPING['portion_array_start_offset'][0]
        max_portion_offset = portion_array_start + (len(portion_list_of_dicts) * SIEMENS_UDT_SIZE_BYTES)
        total_size = max(max_single_offset, max_portion_offset)

        db_data = client.db_read(db_num, 0, total_size)

        # 1. Write single values
        for key, value in single_results_dict.items():
            if key in SIEMENS_TAG_MAPPING:
                offset, dtype, *_ = SIEMENS_TAG_MAPPING[key]
                if dtype == "REAL": set_real(db_data, offset, value)
                elif dtype == "DINT": set_dint(db_data, offset, int(value))
        
        # 2. Write portion array
        for i, portion_dict in enumerate(portion_list_of_dicts):
            if i >= PLC_PORTION_ARRAY_SIZE: break
            
            array_element_start_byte = portion_array_start + (i * SIEMENS_UDT_SIZE_BYTES)
            
            # Now we look up the type for each member of the UDT
            for key, (member_offset, dtype) in SIEMENS_PORTION_UDT_MAP.items():
                if key in portion_dict:
                    byte_offset = array_element_start_byte + member_offset
                    value = portion_dict[key]
                    
                    # Add handlers for each data type in your UDT
                    if dtype == "REAL":
                        set_real(db_data, byte_offset, float(value))
                    elif dtype == "DINT":
                        set_dint(db_data, byte_offset, int(value))

        # 3. Write the entire modified data block back
        client.db_write(db_num, 0, db_data)
        log_func("    ...Successfully wrote all results to Siemens PLC.")
        return True
        
    except Exception as e:
        log_func(f"    ...CRITICAL Siemens PLC Write-Back ERROR: {e}")
        return False
    finally:
        if client.get_connected(): client.disconnect()
    
    
def process_single_file(xyz_file_path, log_messages):
    
    try: 
        print("\n\n--- Headless Pipeline Starting ---")   
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
            PLC_read_starttime = time.time()
            log("    ...PLC_MODE is enabled.")
            plc_values, msg = get_params_from_plc(
                PLC_IP_ADDRESS, 
                PLC_RACK, 
                PLC_SLOT, 
                PLC_DB_NUMBER, 
                SIEMENS_TAG_MAPPING, 
                log
            )  
            if plc_values:
                end_time = time.time()
                log(f"\n--- PLC Read took {end_time - PLC_read_starttime:.2f} seconds ---")
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
            
        # --- Step 1: Load Raw Point Cloud ---
        log(f"\n[1/7] Loading raw data from: {os.path.basename(xyz_file_path)}")
        points_numpy_array = load_point_cloud_from_file(xyz_file_path)
        
        if points_numpy_array is None or points_numpy_array.shape[0] == 0:
            log("Aborting due to file loading error or empty cloud.")
            return
        current_points_df = pd.DataFrame(
            points_numpy_array, columns=['x', 'y', 'z'])
        log(f"    ...Loaded {len(current_points_df)} raw points.")
        
        processed_df = current_points_df.copy()

        # --- Step 2: PCA Alignment (Optional) ---
        if current_pipeline_params.get("pca_align"):
            log("\n[2/7] Applying PCA Alignment...")
            processed_df = align_point_cloud_with_pca(processed_df)
            log("    ...PCA Alignment Complete.")
        else:
            log("\n[2/7] PCA Alignment skipped (disabled in config).")

        # --- Step 3: Auto Downsample (Optional) ---
        if current_pipeline_params.get("enable_auto_downsample") and len(processed_df) > current_pipeline_params.get("auto_downsample_threshold", 9e9):
            threshold = current_pipeline_params.get("auto_downsample_threshold")
            log(f"\n[3/7] Applying Auto-Downsample (Threshold: {threshold:,})...")
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
            log("\n[3/7] Auto-Downsample skipped (not needed or disabled).")

        # --- Step 4: Radius Outlier Removal (Optional) ---
        if current_pipeline_params.get("apply_ror"):
            log("\n[4/7] Applying Radius Outlier Removal Filter...")
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
            log("\n[4/7] ROR Filter skipped (disabled in config).")

        final_display_cloud = processed_df.copy()

        # --- Step 5: Y-Normalization & Final Calculation ---
        log("\n[5/7] Normalizing and Calculating Portions...")
        df_for_calc = processed_df.copy()
        y_offset = 0.0
        if current_pipeline_params.get("enable_y_normalization"):
            if not df_for_calc.empty:
                y_offset = df_for_calc['y'].min()
                df_for_calc['y'] -= y_offset
                log(
                    f"    ...Y-coords normalized for calculation. Scanner Offset: {y_offset:.2f} mm")
        else:
            log("\n[5/7] Normalization skipped (disabled in config).")

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

        # --- Perform Portion Calculation ---
        # If waste redistribution is enabled, use calculate_with_waste_redistribution function
        if current_pipeline_params.get("waste_redistribution"):
            log("    ...Performing Portion Calculation with Waste Redistribution")
            calc_results = calculate_with_waste_redistribution(
            points_df=df_for_calc, verbose_log_func=log, **calculation_args
            )
        else: 
            log("    ...Performing Portion Calculation without Waste Redistribution")
            calc_results = perform_portion_calculation(
                points_df=df_for_calc, verbose_log_func=log, **calculation_args)
         
        if calc_results:
            calc_results["y_offset_for_plot"] = y_offset
        else:
            log("    ...Portion calculation failed or returned no results.")
            return

        log("\n    ...Calculating yield, waste, and length analysis...")
        portions = calc_results.get("portions", [])
        if portions:
            
            # Get the target weight from the parameters used for this run
            target_w = current_pipeline_params.get("target_weight")

            good_weight = 0
            waste_weight = 0
            good_portions_count = 0
            
            # Iterate through each portion and classify it
            for p in portions:
                if p['weight'] >= target_w:
                    good_weight += p['weight']
                    good_portions_count += 1
                else:
                    waste_weight += p['weight']
            
            total_loaf_weight = good_weight + waste_weight
            
            # Avoid division by zero
            yield_percentage = (good_weight / total_loaf_weight * 100) if total_loaf_weight > 0 else 0
            
            product_length = 0.0
            if not final_display_cloud.empty:
                min_y = final_display_cloud['y'].min()
                max_y = final_display_cloud['y'].max()
                product_length = max_y - min_y
            
            # --- Add all new metrics to the results dictionary ---
            calc_results['total_loaf_weight'] = total_loaf_weight
            calc_results['waste_weight'] = waste_weight
            calc_results['good_weight'] = good_weight
            calc_results['good_portions_count'] = good_portions_count
            calc_results['yield_percentage'] = yield_percentage
            calc_results['product_length'] = product_length 
            
            log(f"    ...Product Length: {product_length:.2f}mm, Yield: {yield_percentage:.2f}%, Good Portions: {good_portions_count}, Waste: {waste_weight:.2f}g")
        
         # --- Step 7: Write Results back to Siemens PLC ---
        if PLC_MODE and calc_results:
            PLC_write_starttime = time.time()
            log("\n[6/7] Writing results back to PLC...")
            
            current_counter = current_pipeline_params.get("completion_counter", 0)
            # Increment the counter, with a rollover to prevent overflow
            new_counter = (current_counter + 1) % 2147483647 # Max DINT value
            
            original_portions = calc_results.get("portions", [])
            scanner_offset = calc_results.get("y_offset_for_plot", 0.0)

            # --- A. Prepare single value results ---
            total_good_portions = len(original_portions) - 1 if len(original_portions) > 1 else 0
            first_portion_weight = original_portions[0]['weight'] if original_portions else 0.0
            total_weight_of_good_portions = sum(p['weight'] for p in original_portions[1:])
            avg_weight = total_weight_of_good_portions / total_good_portions if total_good_portions > 0 else 0.0
            density_g_cm3 = calc_results.get("density", 0.0) * 1000.0
                
            single_results_to_write = {
                "completion_counter": new_counter,
                "total_portions_calculated": total_good_portions,
                "average_portion_weight": avg_weight,
                "first_portion_weight": first_portion_weight,
                "calculated_density_g_cm3": density_g_cm3,
                "total_loaf_length": product_length,
                "yield_percentage" : yield_percentage
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
            
            # Call the Siemens-specific write function
            write_results_to_plc(
                PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER,
                SIEMENS_TAG_MAPPING, # Pass the full map for offsets
                SIEMENS_PORTION_UDT_MAP, SIEMENS_UDT_SIZE_BYTES,
                single_results_to_write, 
                portions_to_write, 
                PLC_PORTION_ARRAY_SIZE,
                log
            )

            end_time = time.time()
            log(f"\n--- Pipeline upto PLC Write Time {end_time - start_time:.2f} seconds, Write took {end_time - PLC_write_starttime:.2f} seconds ---")
        else:
            log("\n[6/7] PLC Write-Back skipped (PLC_MODE is disabled or no results).")

        # --- Step 7: Save Payload & Display Results ---
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
            log(f"\n[7/7] Archiving enabled. Saving payload to '{output_path}'...")
        else:
            # Save to the root directory with a unique name
            output_path = unique_filename
            log(f"\n[7/7] Archiving disabled. Saving uniquely named payload to '{output_path}'...")

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

        end_time = time.time()
        print(f"\n--- Pipeline completed in {end_time - start_time:.2f} seconds ---")
        
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
            print("\n--- Headless Pipeline Completed Successfully ---")

        return True, log_messages
    
    except Exception as e:
        log(f"\n--- CRITICAL ERROR in 'process_single_file': {e} ---")
        log(traceback.format_exc())
        return False, log_messages


# In run_pipeline_siemens.py

def start_watcher_service():
    """Main function to run the listener service for a Siemens PLC."""
    print("--- Watcher Service Started (SIEMENS MODE) ---")
    supported_extensions = ('.xyz', '.pcd', '.ply', '.csv', '.xlsx', '.xls')
    print(f"Watching for {', '.join(supported_extensions)} files in: ./{XYZ_INPUT_FOLDER}/")
    print("Press Ctrl+C to stop the service.")

    os.makedirs(XYZ_INPUT_FOLDER, exist_ok=True)
    os.makedirs(XYZ_PROCESSED_FOLDER, exist_ok=True)
    manage_archives(ARCHIVE_BASE_FOLDER, ARCHIVE_CLEANUP_DAYS)
    
    heartbeat_counter = 0
    try:
        # Set initial status to IDLE
        update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, heartbeat_val=heartbeat_counter, status_val=STATUS_IDLE)
        
        while True:
            # Increment and write heartbeat
            heartbeat_counter = (heartbeat_counter + 1) % 2147483647
            update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, heartbeat_val=heartbeat_counter)
            
            found_file = next((f for f in sorted(os.listdir(XYZ_INPUT_FOLDER)) if f.lower().endswith(supported_extensions)), None)
            
            if found_file:
                # File Found: Start Processing
                update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, status_val=STATUS_PROCESSING)
                
                file_path = os.path.join(XYZ_INPUT_FOLDER, found_file)
                success, _ = process_single_file(file_path) # log_capture is handled internally now
                
                # Move the file after processing
                if success:
                    update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, status_val=STATUS_SUCCESS)
                    destination_path = os.path.join(XYZ_PROCESSED_FOLDER, found_file)
                    print(f"--- Moving processed file to: {destination_path} ---")
                    shutil.move(file_path, destination_path)
                else:
                    update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, status_val=STATUS_ERROR)
                    error_folder = os.path.join(XYZ_PROCESSED_FOLDER, "error")
                    os.makedirs(error_folder, exist_ok=True)
                    destination_path = os.path.join(error_folder, found_file)
                    print(f"--- Moving FAILED file to: {destination_path} ---")
                    shutil.move(file_path, destination_path)

                # Set status back to IDLE, ready for the next file
                update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, status_val=STATUS_IDLE)
            else:
                # If no file is found, wait before scanning again
                sys.stdout.write(f"\rNo new files found. Waiting... (Heartbeat: {heartbeat_counter}) ({time.strftime('%H:%M:%S')})")
                sys.stdout.flush()
                time.sleep(WATCHER_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n--- Watcher Service Stopped by User ---")
        update_plc_status(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT, PLC_DB_NUMBER, SIEMENS_TAG_MAPPING, status_val=0)
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---")

if __name__ == "__main__":
    start_watcher_service()
