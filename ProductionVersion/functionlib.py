import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Polygon, MultiPolygon
from sklearn.decomposition import PCA
import alphashape
import sys
import tempfile
import os
import traceback

_open3d_installed = False
try:
    import open3d as o3d
    _open3d_installed = True
except ImportError:
    pass

MIN_POINTS_FOR_HULL = 3
FLOAT_EPSILON = sys.float_info.epsilon
DEFAULT_ALPHA_SHAPE_VALUE = 0.02
DEFAULT_ALPHASHAPE_SLICE_VOXEL = 0.5
DEFAULT_TARGET_WEIGHT = 100.0  # Default target weight in grams
DEFAULT_WEIGHT_TOLERANCE = 0.0
DEFAULT_SLICE_THICKNESS = 0.5


def load_point_cloud_from_file(file_path):
    """
    Loads a point cloud from various file types (XYZ, PCD, PLY, CSV, XLSX, XLS)
    into a NumPy array.

    Args:
        file_path (str): The full path to the point cloud file.

    Returns:
        np.ndarray: An Nx3 NumPy array of points, or None if loading fails.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at '{file_path}'")
        return None

    # Get the file extension and convert to lowercase
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()

    points_array = None
    df = None

    try:
        # --- Open3D Supported Formats (Fastest) ---
        if file_ext in ['.xyz', '.pcd', '.ply']:
            if not _open3d_installed:
                print(f"WARNING: Open3D not installed. Cannot load '{file_ext}' file. Attempting fallback...")
            else:
                pcd = o3d.io.read_point_cloud(file_path)
                if pcd.has_points():
                    points_array = np.asarray(pcd.points)
                    print(f"Successfully loaded {len(points_array)} points from '{file_name}' using Open3D.")
                    return points_array
                else:
                    print(f"WARNING: Open3D read '{file_name}' but it was empty. Attempting pandas fallback for .xyz.")

        # --- Pandas Supported Formats ---
        if file_ext == '.xyz' or file_ext == '.csv':
            # This serves as a fallback for .xyz if Open3D fails, or primary for .csv
            print(f"Attempting to load '{file_name}' using pandas...")
            try:
                # First, try to find 'x', 'y', 'z' columns
                temp_df = pd.read_csv(file_path)
                temp_df.columns = [c.lower() for c in temp_df.columns]
                if {'x', 'y', 'z'}.issubset(temp_df.columns):
                    df = temp_df[['x', 'y', 'z']]
                else:
                    raise ValueError("Named columns not found, trying space-delimited.")
            except (ValueError, UnicodeDecodeError):
                # If that fails, assume it's a simple space-delimited file with no header
                df = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[0, 1, 2], names=['x', 'y', 'z'])

        elif file_ext in ['.xlsx', '.xls']:
            print(f"Attempting to load Excel file '{file_name}'...")
            try:
                # Use openpyxl engine for modern Excel files
                engine = 'openpyxl' if file_ext == '.xlsx' else None
                temp_df = pd.read_excel(file_path, engine=engine)
                temp_df.columns = [str(c).lower() for c in temp_df.columns]
                if {'x', 'y', 'z'}.issubset(temp_df.columns):
                    df = temp_df[['x', 'y', 'z']]
                elif temp_df.shape[1] >= 3:
                    print("WARNING: Columns 'x,y,z' not found. Assuming first 3 columns.")
                    df = temp_df.iloc[:, 0:3]
                    df.columns = ['x', 'y', 'z']
                else:
                    raise ValueError("Not enough columns in Excel file.")
            except ImportError:
                 print("ERROR: Reading Excel files requires 'openpyxl'. Please run 'pip install openpyxl'.")
                 return None
        
        else:
            print(f"ERROR: Unsupported file type '{file_ext}'.")
            return None

        # --- Final Conversion and Validation ---
        if df is not None:
            # Coerce to numeric and drop any rows that failed conversion
            for col in ['x', 'y', 'z']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            
            if not df.empty:
                points_array = df.to_numpy()
                print(f"Successfully loaded {len(points_array)} points from '{file_name}' using pandas.")
            else:
                 print(f"ERROR: No valid numeric XYZ data found in '{file_name}'.")

    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_name}': {e}")
        return None

    return points_array



def align_point_cloud_with_pca(df: pd.DataFrame, shift_to_origin: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    points = df[['x', 'y', 'z']].values

    # 1. Fit and transform using PCA (centers the data)
    pca = PCA(n_components=3)
    aligned_points = pca.fit_transform(points)

    # 2. Re-order columns to match the convention (X=width, Y=length, Z=height)
    final_aligned_points = aligned_points[:, [1, 0, 2]]
    df_aligned = pd.DataFrame(final_aligned_points, columns=['x', 'y', 'z'])

    # 3. --- OPTIONAL ---
    # Shift the entire cloud so its minimum corner is at (0,0,0)
    if shift_to_origin:
        df_aligned['x'] -= df_aligned['x'].min()
        df_aligned['y'] -= df_aligned['y'].min()
        df_aligned['z'] -= df_aligned['z'].min()

    return df_aligned


def estimate_ror_radius_util_o3d(o3d_pcd, k_neighbors, mult):
    if o3d_pcd is None or not o3d_pcd.has_points():
        return None, "Cloud empty for ROR est."
    n_pts = len(o3d_pcd.points)
    if n_pts < k_neighbors + 1:
        return None, f"Not enough pts ({n_pts}) for k={k_neighbors}."
    try:
        tree = o3d.geometry.KDTreeFlann(o3d_pcd)
        dists = []
        #samples = min(500, n_pts)
        samples = min(1000, n_pts)
        indices = np.random.choice(n_pts, size=samples, replace=False)
        for i_idx in indices:  # Renamed i to i_idx
            [k_found, idx_knn, _] = tree.search_knn_vector_3d(  # Renamed idx to idx_knn
                o3d_pcd.points[i_idx], k_neighbors + 1)
            if k_found >= k_neighbors + 1:
                dists.append(np.linalg.norm(
                    o3d_pcd.points[i_idx] - o3d_pcd.points[idx_knn[k_neighbors]]))
        if not dists:
            return None, "No k-th neighbors found. Cloud might be too sparse or k too high."
        avg_dist = np.mean(dists)
        est_rad = avg_dist * mult
        return est_rad, f"Avg k-NN dist ({samples} samples, k={k_neighbors}): {avg_dist:.4f}mm. Suggested Radius (x{mult:.2f}): {est_rad:.4f}mm"
    except Exception as e:
        return None, f"ROR radius est. error: {e}"


def apply_ror_filter_to_df(points_df, ror_nb_points, ror_radius_val, verbose_log_func=None):
    if not _open3d_installed:
        if verbose_log_func:
            verbose_log_func("ROR skipped: Open3D not installed.")
        return points_df, "ROR skipped: Open3D not installed."
    if points_df is None or points_df.empty:
        if verbose_log_func:
            verbose_log_func("ROR skipped: Input DataFrame is empty.")
        return points_df, "ROR skipped: Input DataFrame is empty."
    if not (ror_nb_points > 0 and ror_radius_val > 0):
        if verbose_log_func:
            verbose_log_func(
                f"ROR skipped: Invalid ROR parameters (nb_points={ror_nb_points} or radius={ror_radius_val}). Both must be > 0.")
        return points_df, "ROR skipped: Invalid ROR parameters."

    if verbose_log_func:
        verbose_log_func(
            f"Applying ROR: nb_points={ror_nb_points}, radius={ror_radius_val:.4f}mm")
    pcd_apply_ror = o3d.geometry.PointCloud()
    pcd_apply_ror.points = o3d.utility.Vector3dVector(
        points_df[['x', 'y', 'z']].values)

    if not pcd_apply_ror.has_points():
        if verbose_log_func:
            verbose_log_func("ROR: Cloud is empty before filtering.")
        return points_df, "ROR: Cloud was empty."

    n_before = len(pcd_apply_ror.points)
    try:
        pcd_filtered, _ = pcd_apply_ror.remove_radius_outlier(
            nb_points=int(ror_nb_points), radius=float(ror_radius_val)
        )
        n_after = len(pcd_filtered.points)
        n_removed = n_before - n_after

        if n_after > 0:
            filtered_df = pd.DataFrame(np.asarray(
                pcd_filtered.points), columns=['x', 'y', 'z'])
            status_msg = f"ROR Applied: Removed {n_removed} points. Cloud from {n_before:,} to {n_after:,} points."
            if verbose_log_func:
                verbose_log_func(status_msg)
            return filtered_df, status_msg
        else:
            status_msg = f"ROR Warning: Filter removed all points (from {n_before}). Returning original cloud."
            if verbose_log_func:
                verbose_log_func(status_msg)
            return points_df, status_msg

    except Exception as e_ror_app:
        status_msg = f"ROR Filter Error: {e_ror_app}. Returning original cloud."
        if verbose_log_func:
            verbose_log_func(status_msg)
        return points_df, status_msg


def calculate_slice_profile(
    slice_x_np, slice_z_np,
    flat_bottom, top_down, area_method, alpha_value,
    min_points_for_processing=MIN_POINTS_FOR_HULL,
    alphashape_slice_voxel_param=0.5,
    log_func=None
):
    area = 0.0
    boundary_points_plot = None
    original_point_count_in_slice = len(slice_x_np)
    downsampled_point_count_in_slice = original_point_count_in_slice

    if top_down:
        if len(slice_x_np) >= 2:
            sl_min_x, sl_max_x = slice_x_np.min(), slice_x_np.max()
            width_for_area = sl_max_x - sl_min_x
            mean_z_height = np.mean(slice_z_np)
            if np.isfinite(width_for_area) and np.isfinite(mean_z_height) and width_for_area >= 0 and mean_z_height >= 0:
                area = width_for_area * mean_z_height
            if area > 0:
                boundary_points_plot = np.array([
                    [sl_min_x, 0], [sl_max_x, 0], [sl_max_x, mean_z_height], [
                        sl_min_x, mean_z_height], [sl_min_x, 0]
                ])
        return area, boundary_points_plot, original_point_count_in_slice, downsampled_point_count_in_slice

    current_slice_points_2d = np.column_stack((slice_x_np, slice_z_np))
    if flat_bottom and len(slice_x_np) > 0:
        original_point_count_in_slice = len(current_slice_points_2d)
        downsampled_point_count_in_slice = original_point_count_in_slice

    if len(current_slice_points_2d) < min_points_for_processing:
        return 0.0, None, original_point_count_in_slice, downsampled_point_count_in_slice

    points_for_processing = current_slice_points_2d

    if area_method == "Alpha Shape":
        voxel_size_for_slice = alphashape_slice_voxel_param

        if _open3d_installed and voxel_size_for_slice and voxel_size_for_slice > 0 and len(points_for_processing) > 50:
            try:
                temp_pcd_3d = o3d.geometry.PointCloud()
                temp_pcd_3d.points = o3d.utility.Vector3dVector(
                    np.hstack((points_for_processing, np.zeros(
                        (len(points_for_processing), 1))))
                )
                downsampled_pcd = temp_pcd_3d.voxel_down_sample(
                    voxel_size_for_slice)
                if downsampled_pcd.has_points():
                    points_for_processing = np.asarray(downsampled_pcd.points)[
                        :, :2]
                    downsampled_point_count_in_slice = len(
                        points_for_processing)
            except Exception as e_ds:
                if log_func:
                    log_func(f"Slice downsample error: {e_ds}")
                pass

        if len(points_for_processing) < min_points_for_processing:
            area = 0.0
            boundary_points_plot = None
        else:
            try:
                alpha_polygon_raw = alphashape.alphashape(
                    points_for_processing, alpha_value)

                final_polygon_for_area = None
                if isinstance(alpha_polygon_raw, Polygon):
                    final_polygon_for_area = alpha_polygon_raw
                elif isinstance(alpha_polygon_raw, MultiPolygon):
                    if alpha_polygon_raw.geoms:
                        largest_geom = None
                        max_a = -1.0
                        for geom_item in alpha_polygon_raw.geoms:
                            if geom_item.area > max_a:
                                max_a = geom_item.area
                                largest_geom = geom_item
                        final_polygon_for_area = largest_geom

                if final_polygon_for_area and hasattr(final_polygon_for_area, 'area'):
                    area = final_polygon_for_area.area
                    if hasattr(final_polygon_for_area, 'exterior'):
                        x_coords, z_coords = final_polygon_for_area.exterior.xy
                        boundary_points_plot = np.column_stack(
                            (x_coords, z_coords))
                if not (np.isfinite(area) and area >= 0):
                    area = 0.0
            except Exception as e_alpha:
                area = 0.0
                boundary_points_plot = None
                if log_func:
                    log_func(
                        f"Alpha shape calculation error for slice: {e_alpha}")

    elif area_method == "Convex Hull":
        try:
            hull = ConvexHull(points_for_processing)
            area = hull.volume
            boundary_points_plot = points_for_processing[hull.vertices]
            boundary_points_plot = np.vstack(
                (boundary_points_plot, boundary_points_plot[0]))
            if not (np.isfinite(area) and area >= 0):
                area = 0.0
        except (QhullError, ValueError, Exception):
            area = 0.0
            boundary_points_plot = None

    if area <= FLOAT_EPSILON and len(current_slice_points_2d) >= 2:
        if boundary_points_plot is None:
            eff_min_x = current_slice_points_2d[:, 0].min()
            eff_max_x = current_slice_points_2d[:, 0].max()
            eff_min_z = current_slice_points_2d[:, 1].min()
            eff_max_z = current_slice_points_2d[:, 1].max()
            width_for_area = eff_max_x - eff_min_x
            height_for_area = eff_max_z - eff_min_z
            if width_for_area > FLOAT_EPSILON and height_for_area > FLOAT_EPSILON:
                area = width_for_area * height_for_area
                if area > 0:
                    boundary_points_plot = np.array([
                        [eff_min_x, eff_min_z], [eff_max_x, eff_min_z],
                        [eff_max_x, eff_max_z], [eff_min_x, eff_max_z],
                        [eff_min_x, eff_min_z]
                    ])
            else:
                area = 0.0

    if not (np.isfinite(area) and area >= 0):
        area = 0.0
    return area, boundary_points_plot, original_point_count_in_slice, downsampled_point_count_in_slice


def recalculate_portion_volume(vol_prof, sorted_y_s, slice_inc, p_min_y, p_max_y):
    act_vol = 0.0
    if p_max_y <= p_min_y + FLOAT_EPSILON:
        return 0.0
    s_idx = np.searchsorted(sorted_y_s, p_min_y - slice_inc, side='left')
    e_idx = np.searchsorted(sorted_y_s, p_max_y, side='right')

    for i_idx in range(s_idx, e_idx):  # Renamed i to i_idx
        if i_idx >= len(sorted_y_s):
            break
        ys_c = sorted_y_s[i_idx]
        vol_c = vol_prof.get(ys_c, 0.0)

        if vol_c <= 0:
            continue
        ye_c = sorted_y_s[i_idx+1] if (i_idx+1 <
                                       len(sorted_y_s)) else (ys_c + slice_inc)
        ov_min = max(ys_c, p_min_y)
        ov_max = min(ye_c, p_max_y)
        sl_len_c = ye_c - ys_c

        if ov_max > ov_min + FLOAT_EPSILON:
            if sl_len_c > FLOAT_EPSILON:
                ov_frac = max(0.0, min(1.0, (ov_max - ov_min) / sl_len_c))
                v_add = vol_c * ov_frac
                if np.isfinite(v_add) and v_add >= 0:
                    act_vol += v_add
                else:
                    return np.nan
    return act_vol


def perform_portion_calculation(
    points_df, total_w, target_w, slice_inc, no_interp, flat_bottom, top_down,
    blade_thick, w_tol, start_trim, end_trim, direct_density_g_mm3=None,
    area_calc_method="Convex Hull", alpha_shape_param=DEFAULT_ALPHA_SHAPE_VALUE,
    alphashape_slice_voxel_param=DEFAULT_ALPHASHAPE_SLICE_VOXEL,
    verbose_log_func=None
):
    calc_t_start = time.time()
    tmp_portions = []
    out_portions = []
    tot_vol = 0.0
    dens = 0.0
    vp_time = 0.0
    cut_time = 0.0
    vp_prof = {}
    res = {"portions": [], "total_volume": 0.0, "density": 0.0, "status": "Init Err", "calc_time": 0.0,
           "volume_profile": {}, "sorted_y_starts": np.array([]), "calc_start_y": 0.0, "calc_end_y": 0.0,
           "y_offset_for_plot": 0.0, "density_source_message": ""}

    log = verbose_log_func if verbose_log_func else lambda x, end_line=True: None

    log("Portion calculation: Starting...")

    if points_df is None or points_df.empty:
        res["status"] = "Err: No cloud data for calculation."
        log(res["status"])
        return res
    if not np.all(np.isfinite(points_df[['x', 'y', 'z']].values)):
        res["status"] = "Err: Cloud has non-finite values."
        log(res["status"])
        return res

    loaf_min_y_orig_calc, loaf_max_y_orig_calc = points_df['y'].min(
    ), points_df['y'].max()
    if loaf_max_y_orig_calc - loaf_min_y_orig_calc <= FLOAT_EPSILON:
        res["status"] = "Err: Loaf length (for calc) near zero."
        log(res["status"])
        return res

    calc_sy_eff = loaf_min_y_orig_calc + start_trim
    calc_ey_eff = loaf_max_y_orig_calc - end_trim
    res["calc_start_y"] = calc_sy_eff
    res["calc_end_y"] = calc_ey_eff

    if calc_sy_eff >= calc_ey_eff - FLOAT_EPSILON:
        res["status"] = f"Err: Trims overlap (Effective calc length: {calc_ey_eff-calc_sy_eff:.2f}mm)."
        log(res["status"])
        return res

    try:
        _vp_start_t = time.time()
        log("Preparing point cloud (sorting by Y)...")
        points_df_sorted = points_df.sort_values(
            by='y', kind='mergesort', ignore_index=True)
        y_coords_np = points_df_sorted['y'].to_numpy(
            dtype=np.float64, copy=False)
        x_coords_np = points_df_sorted['x'].to_numpy(
            dtype=np.float64, copy=False)
        z_coords_np = points_df_sorted['z'].to_numpy(
            dtype=np.float64, copy=False)
        log("Point cloud prepared for profiling.")

        full_y_steps = np.arange(loaf_min_y_orig_calc,
                                 loaf_max_y_orig_calc, slice_inc)
        n_steps_vp = len(full_y_steps) if len(full_y_steps) > 0 else 1

        for i_vp, y_s_loop in enumerate(full_y_steps):  # Renamed i to i_vp
            if verbose_log_func and (i_vp % max(1, n_steps_vp // 20) == 0 or i_vp == n_steps_vp - 1):
                prog_msg_vp = f"Vol profile... Slice {i_vp+1}/{n_steps_vp} (Y={y_s_loop:.1f}mm)"
                log(f"\r{prog_msg_vp}      ", end_line=False)

            y_e_loop = min(y_s_loop + slice_inc, loaf_max_y_orig_calc)
            if y_s_loop >= loaf_max_y_orig_calc - FLOAT_EPSILON:
                break

            start_idx = np.searchsorted(y_coords_np, y_s_loop, side='left')
            end_idx = np.searchsorted(y_coords_np, y_e_loop, side='left')

            sl_area = 0.0
            if start_idx < end_idx:
                slice_x_for_profile = x_coords_np[start_idx:end_idx]
                slice_z_for_profile = z_coords_np[start_idx:end_idx]

                slice_log_func = None
                if verbose_log_func:
                    def slice_log_func(msg_sl): return log(
                        f"SliceProfileLOG (Y={y_s_loop:.1f}): {msg_sl}")

                sl_area, _, _, _ = calculate_slice_profile(
                    slice_x_for_profile, slice_z_for_profile,
                    flat_bottom if not top_down else False,
                    top_down, area_calc_method, alpha_shape_param,
                    alphashape_slice_voxel_param=alphashape_slice_voxel_param,
                    log_func=slice_log_func
                )

            slice_actual_length = max(0.0, y_e_loop - y_s_loop)
            sl_vol = sl_area * slice_actual_length
            if sl_vol > 0:
                vp_prof[y_s_loop] = sl_vol
                tot_vol += sl_vol
        if verbose_log_func:
            log("", end_line=True)

        vp_time = time.time() - _vp_start_t
        log(f"Vol profile done ({vp_time:.2f}s). Est. Total Vol: {tot_vol/1000.0:.1f} cm³.")

        if direct_density_g_mm3 is not None and direct_density_g_mm3 > FLOAT_EPSILON:
            dens = direct_density_g_mm3
            res["density_source_message"] = f"Using direct density: {dens * 1000.0:.3f} g/cm³."
        else:
            if tot_vol <= FLOAT_EPSILON * 100:
                raise ValueError(
                    f"Est. total volume ({tot_vol:.2e}mm³) is too small. Check cloud/params.")
            if total_w <= FLOAT_EPSILON:
                raise ValueError(
                    "Total weight must be >0 to calculate density from volume.")
            dens = total_w / tot_vol
            res["density_source_message"] = f"Calculated density: {dens * 1000.0:.3f} g/cm³ (Wt: {total_w:.2f}g / Vol: {tot_vol/1000.0:.1f}cm³). VP Time: {vp_time:.2f}s"

        if not (np.isfinite(dens) and dens > FLOAT_EPSILON):
            raise ValueError(f"Density invalid ({dens=}).")

        res["total_volume"] = tot_vol
        res["density"] = dens
        sorted_y_s_arr = np.array(sorted(vp_prof.keys()))
        res["sorted_y_starts"] = sorted_y_s_arr
        res["volume_profile"] = vp_prof

        _cut_start_t = time.time()
        current_w_accum = 0.0
        last_phys_cut_y = calc_ey_eff

        relevant_slice_starts_mask = (sorted_y_s_arr < calc_ey_eff) & \
                                     ((sorted_y_s_arr + slice_inc) > calc_sy_eff)
        relevant_y_s_arr = sorted_y_s_arr[relevant_slice_starts_mask]

        if len(relevant_y_s_arr) == 0:
            raise ValueError("No profile slices in trim range for cutting.")

        y_s_rev_iter = relevant_y_s_arr[::-1]
        target_w_lower = target_w - w_tol
        n_rev_sl_cut = len(y_s_rev_iter)

        log(
            f"Starting reversed cutting from Y={last_phys_cut_y:.2f}mm down to Y={calc_sy_eff:.2f}mm...")

        for rev_i, y_sl_s_abs in enumerate(y_s_rev_iter):
            if verbose_log_func and (rev_i % max(1, n_rev_sl_cut // 10) == 0 or rev_i == n_rev_sl_cut - 1):
                prog_msg_cut = f"Cutting (rev)... Slice at Y={y_sl_s_abs:.1f}mm ({rev_i+1}/{n_rev_sl_cut})"
                log(f"\r{prog_msg_cut}     ", end_line=False)

            y_sl_e_abs = y_sl_s_abs + slice_inc
            eval_part_s = max(y_sl_s_abs, calc_sy_eff)
            eval_part_e = min(y_sl_e_abs, last_phys_cut_y)

            if eval_part_e <= eval_part_s + FLOAT_EPSILON:
                continue

            vol_full_sl = vp_prof.get(y_sl_s_abs, 0.0)
            if vol_full_sl <= FLOAT_EPSILON:
                continue

            eval_part_len = eval_part_e - eval_part_s
            frac_in_eval = eval_part_len / slice_inc if slice_inc > FLOAT_EPSILON else 0.0
            frac_in_eval = max(0.0, min(1.0, frac_in_eval))

            wt_of_eval_part = (vol_full_sl * frac_in_eval) * dens

            if not (np.isfinite(wt_of_eval_part) and wt_of_eval_part >= 0):
                wt_of_eval_part = 0.0

            if current_w_accum + wt_of_eval_part >= target_w_lower - FLOAT_EPSILON:
                cut_loc_y = eval_part_s
                wt_carry_over = 0.0

                if not no_interp:
                    needed_w = target_w - current_w_accum
                    if wt_of_eval_part > FLOAT_EPSILON and needed_w > 0 and needed_w < wt_of_eval_part:
                        frac_to_take = needed_w / wt_of_eval_part
                        frac_to_take = max(0.0, min(1.0, frac_to_take))
                        cut_loc_y = eval_part_e - \
                            (frac_to_take * eval_part_len)
                        wt_carry_over = wt_of_eval_part * (1.0 - frac_to_take)
                    elif needed_w <= 0:
                        cut_loc_y = eval_part_e
                        wt_carry_over = wt_of_eval_part

                cut_loc_y = max(cut_loc_y, calc_sy_eff)

                p_max_y_abs = last_phys_cut_y
                p_min_y_abs = cut_loc_y
                p_len = max(0.0, p_max_y_abs - p_min_y_abs)
                act_p_wt = 0.0

                if p_len > FLOAT_EPSILON:
                    act_p_vol = recalculate_portion_volume(
                        vp_prof, sorted_y_s_arr, slice_inc, p_min_y_abs, p_max_y_abs)
                    if np.isnan(act_p_vol):
                        raise ValueError(
                            f"Invalid volume in portion recalc (ends {p_max_y_abs:.2f}).")
                    act_p_wt = act_p_vol * dens

                tmp_portions.append({
                    "calc_max_y": p_max_y_abs, "calc_min_y": p_min_y_abs,
                    "length": p_len, "weight": act_p_wt if np.isfinite(act_p_wt) else 0.0
                })

                last_phys_cut_y = p_min_y_abs - blade_thick
                current_w_accum = wt_carry_over if p_min_y_abs > calc_sy_eff + FLOAT_EPSILON else 0.0
            else:
                current_w_accum += wt_of_eval_part

            if last_phys_cut_y <= calc_sy_eff + FLOAT_EPSILON:
                break
        if verbose_log_func:
            log("", end_line=True)

        cut_time = time.time() - _cut_start_t

        if last_phys_cut_y > calc_sy_eff + FLOAT_EPSILON:
            w_max_y = last_phys_cut_y
            w_min_y = calc_sy_eff
            w_len = max(0.0, w_max_y - w_min_y)
            w_wt = 0.0
            if w_len > FLOAT_EPSILON:
                w_vol = recalculate_portion_volume(
                    vp_prof, sorted_y_s_arr, slice_inc, w_min_y, w_max_y)
                if np.isnan(w_vol):
                    raise ValueError("Invalid volume for final waste.")
                w_wt = w_vol * dens

            tmp_portions.append({
                "calc_max_y": w_max_y, "calc_min_y": w_min_y,
                "length": w_len, "weight": w_wt if np.isfinite(w_wt) else 0.0
            })

        final_ordered_ps = tmp_portions[::-1]
        # Renamed i to i_portion
        for i_portion, p_data in enumerate(final_ordered_ps):
            pmy, pmxy = p_data.get("calc_min_y"), p_data.get("calc_max_y")
            out_portions.append({
                "portion_num": i_portion + 1,
                "display_start_y": pmy if np.isfinite(pmy) else 0.0,
                "display_end_y": pmxy if np.isfinite(pmxy) else 0.0,
                "length": p_data.get("length", 0.0),
                "weight": p_data.get("weight", 0.0),
                "cut_y": pmxy if np.isfinite(pmxy) else 0.0
            })

        stat_msg = f"Calc complete. Found {len(out_portions)} portions."
        if out_portions:
            stat_msg += " (P1 is first physical piece/waste)."

    except OverflowError:
        stat_msg = "Err: Numerical overflow."
    except ValueError as ve:
        stat_msg = f"Err: {ve}"
    except Exception as ex:
        stat_msg = f"Unexpected err: {ex}"
        import traceback
        if verbose_log_func:
            log(traceback.format_exc())
    finally:
        tot_calc_t = time.time() - calc_t_start
        final_stat_msg = f"{stat_msg} Total Time: {tot_calc_t:.2f}s (Profile: {vp_time:.2f}s, Cutting: {cut_time:.2f}s)"
        log(final_stat_msg)

        res["portions"] = out_portions
        res["status"] = final_stat_msg
        res["calc_time"] = tot_calc_t

    return res


def plot_3d_loaf(points_df, portions=None, title="Point Cloud", y_offset=0.0, camera_override=None, selected_colorscale='YlOrBr'):
    if points_df is None or points_df.empty:
        return go.Figure()
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points_df['x'], y=points_df['y'], z=points_df['z'],
        mode='markers',
        marker=dict(
            size=1.5,
            color=points_df['z'],
            colorscale=selected_colorscale,
            opacity=0.7,
            colorbar=dict(title='Height (Z)'),
            showscale=False
        ),
        name='Point Cloud'
    ))

    if portions and len(portions) > 0:
        min_x_plot, max_x_plot = points_df['x'].min(), points_df['x'].max()
        x_rng_plot = max(1.0, max_x_plot - min_x_plot)
        min_z_plot, max_z_plot = points_df['z'].min(), points_df['z'].max()
        z_rng_plot = max(1.0, max_z_plot - min_z_plot)

        # Renamed i to i_plot, p to p_plot_data
        for i_plot, p_plot_data in enumerate(portions):
            cut_y_val = p_plot_data.get('display_end_y')

            if cut_y_val is not None and i_plot < len(portions) - 1:
                fig.add_trace(go.Mesh3d(
                    x=[min_x_plot - 0.1*x_rng_plot, max_x_plot + 0.1*x_rng_plot,
                        max_x_plot + 0.1*x_rng_plot, min_x_plot - 0.1*x_rng_plot],
                    y=[cut_y_val, cut_y_val, cut_y_val, cut_y_val],
                    z=[min_z_plot - 0.1*z_rng_plot, min_z_plot - 0.1*z_rng_plot,
                        max_z_plot + 0.1*z_rng_plot, max_z_plot + 0.1*z_rng_plot],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    opacity=0.3,
                    color='red',
                    name=f'Cut after P{p_plot_data.get("portion_num")}',
                    showlegend=(True)
                ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Width (X,mm)',
            yaxis_title='Length (Y,mm)',
            zaxis_title='Height (Z,mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    if camera_override:
        fig.update_layout(scene_camera=camera_override)
    return fig


def plot_area_profile(volume_profile, sorted_y_starts, slice_increment_mm, calc_start_y=None, calc_end_y=None, y_offset_for_plot=0.0):
    if not volume_profile or len(sorted_y_starts) == 0:
        return go.Figure(layout=dict(title="Area Profile (No Data)", height=300))

    y_values_plot = sorted_y_starts + y_offset_for_plot
    area_values_plot = [(volume_profile.get(y_val_calc, 0) / slice_increment_mm if slice_increment_mm > FLOAT_EPSILON else 0)
                        for y_val_calc in sorted_y_starts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_values_plot, y=area_values_plot, mode='lines',
                  name='Est. Area', line=dict(shape='spline', smoothing=0.5)))

    plot_min_y_boundary = y_values_plot[0] if len(y_values_plot) > 0 else (
        calc_start_y if calc_start_y is not None else 0)
    plot_max_y_boundary = (y_values_plot[-1] + slice_increment_mm) if len(
        y_values_plot) > 0 else (calc_end_y if calc_end_y is not None else 0)

    if calc_start_y is not None and calc_start_y > (sorted_y_starts[0] + y_offset_for_plot if len(sorted_y_starts) > 0 else -np.inf):
        fig.add_vrect(x0=plot_min_y_boundary, x1=calc_start_y, fillcolor="grey",
                      opacity=0.15, layer="below", line_width=0, name="Start Trim")
    if calc_end_y is not None and calc_end_y < ((sorted_y_starts[-1] + slice_increment_mm) + y_offset_for_plot if len(sorted_y_starts) > 0 else np.inf):
        fig.add_vrect(x0=calc_end_y, x1=plot_max_y_boundary, fillcolor="grey",
                      opacity=0.15, layer="below", line_width=0, name="End Trim")

    fig.update_layout(
        title="Area Profile (Shaded=Trimmed Sections, Y-axis uses original coords)",
        xaxis_title=f"Length (Y, mm - Original Coords)",
        yaxis_title="Est. Cross-Sectional Area (mm²)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return fig


def plot_cumulative_weight(volume_profile, sorted_y_starts, density_g_mm3, portions, target_weight, tolerance, calc_start_y=None, calc_end_y=None, y_offset_for_plot=0.0, slice_increment_mm=0.5):
    if not volume_profile or len(sorted_y_starts) == 0 or density_g_mm3 <= 0:
        return go.Figure(layout=dict(title="Cumulative Weight (No Data)", height=350))

    y_coords_plot_calc = []
    cumulative_weights_plot = []
    current_weight_plot = 0.0

    if sorted_y_starts.size > 0:
        y_coords_plot_calc.append(sorted_y_starts[0])
        cumulative_weights_plot.append(0.0)

        # Renamed i to i_plot_weight
        for i_plot_weight, y_s_plot_calc in enumerate(sorted_y_starts):
            slice_vol_plot = volume_profile.get(y_s_plot_calc, 0.0)
            current_weight_plot += (slice_vol_plot * density_g_mm3)
            y_e_plot_calc = sorted_y_starts[i_plot_weight+1] if (i_plot_weight + 1 < len(
                sorted_y_starts)) else (y_s_plot_calc + slice_increment_mm)
            y_coords_plot_calc.append(y_e_plot_calc)
            cumulative_weights_plot.append(current_weight_plot)

    y_coords_plot_display = [y + y_offset_for_plot for y in y_coords_plot_calc]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_coords_plot_display, y=cumulative_weights_plot,
                  mode='lines', name='Cumulative Weight'))
    fig.add_hline(y=target_weight, line_dash="dash",
                  line_color="red", name="Target Weight")
    if tolerance > 0:
        fig.add_hline(y=target_weight + tolerance, line_dash="dot",
                      line_color="orange", name="Target +Tolerance")
        fig.add_hline(y=target_weight - tolerance, line_dash="dot",
                      line_color="orange", name="Target -Tolerance")

    max_weight_for_plot = max(
        cumulative_weights_plot) if cumulative_weights_plot else target_weight * 1.1

    # Renamed i to i_portion_plot, p_plot to p_plot_item
    for i_portion_plot, p_plot_item in enumerate(portions):
        cut_y_plot_loc = p_plot_item.get('display_end_y', np.nan)
        if np.isfinite(cut_y_plot_loc) and i_portion_plot < len(portions) - 1:
            if p_plot_item.get('length', 0) > FLOAT_EPSILON:
                fig.add_vline(x=cut_y_plot_loc, line_dash="solid", line_color="grey",
                              name=f"End of P{i_portion_plot+1}" if i_portion_plot < 3 else None, showlegend=(i_portion_plot < 3 and i_portion_plot < len(portions) - 1))

    plot_min_y_boundary = y_coords_plot_display[0] if y_coords_plot_display else (
        calc_start_y if calc_start_y is not None else 0)
    plot_max_y_boundary = y_coords_plot_display[-1] if y_coords_plot_display else (
        calc_end_y if calc_end_y is not None else 0)

    if calc_start_y is not None and calc_start_y > (sorted_y_starts[0] + y_offset_for_plot if len(sorted_y_starts) > 0 else -np.inf):
        fig.add_vrect(x0=plot_min_y_boundary, x1=calc_start_y, fillcolor="grey",
                      opacity=0.15, layer="below", line_width=0, name="Start Trim")
    if calc_end_y is not None and calc_end_y < ((sorted_y_starts[-1] + slice_increment_mm) + y_offset_for_plot if len(sorted_y_starts) > 0 else np.inf):
        fig.add_vrect(x0=calc_end_y, x1=plot_max_y_boundary, fillcolor="grey",
                      opacity=0.15, layer="below", line_width=0, name="End Trim")

    fig.update_layout(title="Cumulative Weight Profile (Shaded=Trimmed, Y-axis uses original coords)",
                      xaxis_title="Length (Y, mm - Original Coords)",
                      yaxis_title="Cumulative Weight (g)", yaxis_range=[0, max_weight_for_plot * 1.05],
                      margin=dict(l=20, r=20, t=40, b=20), height=350)
    return fig


o3d_animation_globals = {
    "current_step": 0, "total_steps": 600, "animation_type": None,
    "pcd_center": np.array([0, 0, 0]), "initial_view_set": False
}


def reset_o3d_animation_globals():
    o3d_animation_globals["current_step"] = 0
    o3d_animation_globals["animation_type"] = None
    o3d_animation_globals["initial_view_set"] = False


def o3d_animation_callback(vis):
    state = o3d_animation_globals
    view_control = vis.get_view_control()
    if not state["initial_view_set"]:
        view_control.set_zoom(0.7)
        view_control.set_lookat(state["pcd_center"])
        state["initial_view_set"] = True
    if state["current_step"] >= state["total_steps"]:
        return False
    if state["animation_type"] == 'o3d_fly_around':
        view_control.rotate(5.0, 0.0)
    state["current_step"] += 1
    return True


def start_o3d_visualization(points_df, animation_type_str='o3d_fly_around'):
    if not _open3d_installed:
        return
    if points_df is None or points_df.empty:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_df[['x', 'y', 'z']].values)
    if not pcd.has_points():
        return

    reset_o3d_animation_globals()
    o3d_animation_globals["animation_type"] = animation_type_str
    o3d_animation_globals["pcd_center"] = pcd.get_center()
    o3d_animation_globals["total_steps"] = 600

    window_title = "Open3D Fly Around"
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1280, height=720)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 2.5
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        vis.register_animation_callback(o3d_animation_callback)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Failed to launch Open3D window: {e}")
    finally:
        print("Open3D window closed.")
        reset_o3d_animation_globals()


def create_o3d_plane_mesh(y_pos, x_min, x_max, z_min, z_max, color, plane_thickness_along_y=1.0):
    if not _open3d_installed:
        return None
    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=(x_max - x_min), height=plane_thickness_along_y, depth=(z_max - z_min))
    mesh_box.translate([x_min, y_pos - (plane_thickness_along_y / 2), z_min])
    mesh_box.paint_uniform_color(color)
    return mesh_box


def launch_o3d_viewer_with_cuts(points_df, portions_data, calc_start_y_from_res, calc_end_y_from_res, y_offset_to_add_back=0.0):
    # y_offset_to_add_back is no longer strictly needed here as coordinates should be pre-adjusted
    if not _open3d_installed:
        print("Open3D library is not installed.")
        return
    if points_df is None or points_df.empty:
        print("No point cloud data for Open3D cuts view.")
        return
    if not portions_data:
        print("No portion data for Open3D cuts view.")
        return

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(
        points_df[['x', 'y', 'z']].values)
    if not pcd_o3d.has_points():
        print("Point cloud empty for Open3D visualization.")
        return

    geometries = [pcd_o3d]
    x_min_viz, x_max_viz = points_df['x'].min(), points_df['x'].max()
    z_min_viz, z_max_viz = points_df['z'].min(), points_df['z'].max()
    x_pad, z_pad = (x_max_viz - x_min_viz)*0.05, (z_max_viz - z_min_viz)*0.05
    plane_x_min, plane_x_max = x_min_viz - x_pad, x_max_viz + x_pad
    plane_z_min, plane_z_max = z_min_viz - z_pad, z_max_viz + z_pad

    color_portion_cut, color_trim_cut = [0.8, 0.2, 0.2], [0.2, 0.2, 0.8]
    plane_thick = max(
        1.0, (points_df['y'].max() - points_df['y'].min()) * 0.002)

    # calc_start_y_from_res and calc_end_y_from_res are already in original coordinates
    if calc_start_y_from_res is not None:
        start_plane = create_o3d_plane_mesh(calc_start_y_from_res,
                                            plane_x_min, plane_x_max, plane_z_min, plane_z_max, color_trim_cut, plane_thick)
        if start_plane:
            geometries.append(start_plane)

    if calc_end_y_from_res is not None:
        end_plane = create_o3d_plane_mesh(calc_end_y_from_res,
                                          plane_x_min, plane_x_max, plane_z_min, plane_z_max, color_trim_cut, plane_thick)
        if end_plane:
            geometries.append(end_plane)

    # Renamed i to i_viewer_cuts
    for i_viewer_cuts, p_item in enumerate(portions_data):
        # 'display_end_y' from portion data is already in original coordinates
        cut_y_val_display = p_item.get('display_end_y')

        if cut_y_val_display is not None:
            is_s_trim = (calc_start_y_from_res is not None and abs(
                cut_y_val_display - calc_start_y_from_res) < FLOAT_EPSILON)
            is_e_trim = (calc_end_y_from_res is not None and abs(
                cut_y_val_display - calc_end_y_from_res) < FLOAT_EPSILON)

            # Only draw portion cuts if they are not trim cuts and not the end of the very last piece
            if not is_s_trim and not is_e_trim and i_viewer_cuts < len(portions_data) - 1:
                portion_plane = create_o3d_plane_mesh(cut_y_val_display,
                                                      plane_x_min, plane_x_max, plane_z_min, plane_z_max, color_portion_cut, plane_thick)
                if portion_plane:
                    geometries.append(portion_plane)

    print("Launching Open3D viewer with cuts. Close window to resume.")
    try:
        o3d.visualization.draw_geometries(geometries, window_name="Open3D - Point Cloud with Cuts",
                                          width=1280, height=720, point_show_normal=False)
    except Exception as e:
        print(f"Failed to launch Open3D window: {e}")
    finally:
        print("Open3D window closed.")    


def calculate_with_waste_redistribution(
    points_df, total_w, target_w, slice_inc,
    start_trim, end_trim, blade_thick,
    max_target_weight_increase_percent=99.0,
    direct_density_g_mm3=None,
    verbose_log_func=None,
    **kwargs
):

    calc_t_start = time.time()
    log = verbose_log_func if verbose_log_func else lambda msg, end_line=True: None
    
    res = {
        "portions": [], "total_volume": 0.0, "density": 0.0, "status": "Init Err",
        "calc_time": 0.0, "volume_profile": {}, "sorted_y_starts": np.array([]),
        "calc_start_y": 0.0, "calc_end_y": 0.0, "y_offset_for_plot": 0.0,
        "density_source_message": "", "optimization_log": []
    }

    log("Portion calculation (Waste Redistribution - Cumulative Weight): Starting...")
    
    # --- 1. Initial Checks ---
    if points_df is None or points_df.empty:
        res["status"] = "Err: No cloud data for calculation."; log(res["status"]); return res
    
    loaf_min_y, loaf_max_y = points_df['y'].min(), points_df['y'].max()
    calc_start_y = loaf_min_y + start_trim
    calc_end_y = loaf_max_y - end_trim
    res.update({"calc_start_y": calc_start_y, "calc_end_y": calc_end_y})

    if calc_start_y >= calc_end_y - FLOAT_EPSILON:
        res["status"] = f"Err: Start trim ({start_trim}mm) is beyond loaf length."; log(res["status"]); return res

    try:
        # --- 2. Volume Profiling ---
        _vp_start_t = time.time()
        log("Preparing point cloud and profiling volume...")
        
        points_df_sorted = points_df.sort_values(by='y', kind='mergesort', ignore_index=True)
        y_coords_np = points_df_sorted['y'].to_numpy()
        x_coords_np = points_df_sorted['x'].to_numpy()
        z_coords_np = points_df_sorted['z'].to_numpy()
        
        y_steps = np.arange(loaf_min_y, loaf_max_y, slice_inc)
        vp_prof = {}

        slice_profile_args = {
            'flat_bottom': kwargs.get('flat_bottom', False), 'top_down': kwargs.get('top_down_scan', False),
            'area_method': kwargs.get('area_calc_method', 'Convex Hull'), 'alpha_value': kwargs.get('alpha_shape_param', 0.02),
            'alphashape_slice_voxel_param': kwargs.get('alphashape_slice_voxel_param', 0.5)
        }
        
        for y_s in y_steps:
            y_e = y_s + slice_inc
            start_idx, end_idx = np.searchsorted(y_coords_np, [y_s, y_e], side='left')
            if start_idx < end_idx:
                area, _, _, _ = calculate_slice_profile(
                    x_coords_np[start_idx:end_idx], z_coords_np[start_idx:end_idx], **slice_profile_args
                )
                vp_prof[y_s] = area * slice_inc
        
        tot_vol = sum(vp_prof.values())
        vp_time = time.time() - _vp_start_t
        log(f"Volume profile done ({vp_time:.2f}s). Est. Total Vol: {tot_vol/1000.0:.1f} cm³.")
        
        # --- 3. Density Calculation ---
        if direct_density_g_mm3:
            dens = direct_density_g_mm3
            res["density_source_message"] = f"Using direct density: {dens * 1000.0:.3f} g/cm³."
        else:
            if tot_vol <= 0: raise ValueError("Total volume is zero.")
            if total_w <= 0: raise ValueError("Total weight must be >0 to calculate density.")
            dens = total_w / tot_vol
            res["density_source_message"] = f"Calculated density: {dens * 1000.0:.3f} g/cm³."
        
        res.update({"total_volume": tot_vol, "density": dens, "volume_profile": vp_prof})
        sorted_y_s_arr = np.array(sorted(vp_prof.keys()))
        res["sorted_y_starts"] = sorted_y_s_arr
        
        # --- 4. Waste Redistribution Logic ---
        opt_log = []
        
        total_usable_volume = recalculate_portion_volume(vp_prof, sorted_y_s_arr, slice_inc, calc_start_y, calc_end_y)
        total_usable_weight = total_usable_volume * dens
        opt_log.append(f"Total usable loaf weight (after trims): {total_usable_weight:.2f}g")

        if total_usable_weight < target_w:
            raise ValueError("Total usable weight is less than a single target portion.")
        
        num_portions_to_make = int(total_usable_weight // target_w)
        opt_log.append(f"Can ideally make {num_portions_to_make} portions of at least {target_w:.2f}g.")

        if num_portions_to_make == 0:
            raise ValueError("Cannot make any portions of the target weight.")
        
        new_target_w = total_usable_weight / num_portions_to_make
        max_allowed_target_w = target_w * (1 + max_target_weight_increase_percent / 100.0)
        
        final_target_w = target_w
        if new_target_w > max_allowed_target_w:
            log(f"Redistribution aborted: New target ({new_target_w:.2f}g) exceeds max allowed ({max_allowed_target_w:.2f}g).")
            opt_log.append(f"Calculated new target {new_target_w:.2f}g exceeded limit of {max_allowed_target_w:.2f}g. Reverting to original target.")
            res['optimized_target_weight'] = None
        else:
            log(f"Optimal new target weight calculated: {new_target_w:.2f}g to maximize yield.")
            opt_log.append(f"Optimal new target weight: {new_target_w:.2f}g (within {max_target_weight_increase_percent}% limit).")
            final_target_w = new_target_w
            res['optimized_target_weight'] = final_target_w

        res["optimization_log"] = opt_log
        
        # --- 5. Final Cutting Logic using Cumulative Weight Profile ---
        _cut_start_t = time.time()
        out_portions = []
        
        # a. Create the cumulative weight profile for fast lookups
        cumulative_y = [calc_start_y]
        cumulative_w = [0.0]
        
        relevant_slices_mask = (sorted_y_s_arr >= calc_start_y - slice_inc) & (sorted_y_s_arr < calc_end_y)
        for y_s in sorted_y_s_arr[relevant_slices_mask]:
            slice_weight = vp_prof.get(y_s, 0.0) * dens
            cumulative_w.append(cumulative_w[-1] + slice_weight)
            cumulative_y.append(y_s + slice_inc)
        
        cum_w_np = np.array(cumulative_w)
        cum_y_np = np.array(cumulative_y)

        # b. Find cut points by searching the cumulative profile
        current_y = calc_start_y
        current_w = 0.0

        for i in range(num_portions_to_make):
            # For the last portion, the cut is simply at the very end
            if i == num_portions_to_make - 1:
                cut_y = calc_end_y
            else:
                # Find the Y-coordinate where the cumulative weight equals our target
                target_cumulative_weight = current_w + final_target_w
                
                # Use np.interp to find the exact Y for the target weight.
                # This is extremely fast and accurate.
                cut_y = np.interp(target_cumulative_weight, cum_w_np, cum_y_np)

            cut_y = min(cut_y, calc_end_y)
            p_len = cut_y - current_y
            
            if p_len <= FLOAT_EPSILON: continue

            # Recalculate the exact volume for this portion for maximum accuracy
            p_vol = recalculate_portion_volume(vp_prof, sorted_y_s_arr, slice_inc, current_y, cut_y)
            p_weight = p_vol * dens
            
            out_portions.append({
                "portion_num": i + 1, "display_start_y": current_y,
                "display_end_y": cut_y, "length": p_len, "weight": p_weight
            })

            # Update our position for the next search
            current_y = cut_y + blade_thick
            # Find the new starting weight by searching for the new start_y
            current_w = np.interp(current_y, cum_y_np, cum_w_np)

        cut_time = time.time() - _cut_start_t
        res["portions"] = out_portions
        res["status"] = f"Calculation with Waste Redistribution complete. Found {len(out_portions)} portions."

    except Exception as ex:
        res["status"] = f"Unexpected err: {ex}"
        log(traceback.format_exc())
    finally:
        tot_calc_t = time.time() - calc_t_start
        vp_time_str = f", Profile: {vp_time:.2f}s" if '_vp_start_t' in locals() else ""
        cut_time_str = f", Cutting: {cut_time:.2f}s" if '_cut_start_t' in locals() else ""
        res["calc_time"] = tot_calc_t
        res["status"] += f" Total Time: {tot_calc_t:.2f}s{vp_time_str}{cut_time_str}"
        log(res["status"])
        
    return res
