# -----------------------------------------------------------------------------
# Cheese Loaf Portion Calculator (Streamlit App)
# Version: 2.6 (Features: File Uploads (CSV/XYZ/PCD), Test Data Gen,
#                 Volume Modes (Hull, Flat Bottom, Top-Down), Reverse Calc,
#                 Waste First, Interpolation Toggle, Kerf, Tolerance, Trims,
#                 Downsampling, Plotting (3D + 2D Profiles))
# -----------------------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from scipy.spatial import ConvexHull, QhullError # For volume calculation
import sys    # For float info
import time   # For basic timing
import tempfile # For temporary file handling with open3d
import os       # For temporary file handling

# --- Optional: Import open3d with error handling ---
_open3d_installed = False
try:
    import open3d as o3d
    _open3d_installed = True
except ImportError:
    # Warning will be shown in the sidebar if needed later
    pass
# --- End Optional ---

# --- Constants and Configuration ---
NOMINAL_HEIGHT = 90.0  # mm
NOMINAL_WIDTH = 93.3   # mm
NOMINAL_LENGTH = 360.0 # mm
DEFAULT_TARGET_WEIGHT = 250.0 # g
DEFAULT_TOTAL_WEIGHT = (NOMINAL_WIDTH * NOMINAL_HEIGHT * NOMINAL_LENGTH) * 1.05e-9 * 1050 * 1000 # Approx g
DEFAULT_RESOLUTION = 0.5 # mm
DEFAULT_SLICE_THICKNESS = 0.5 # mm
MIN_POINTS_FOR_HULL = 10 # Min points in slice for convex hull attempt
FLOAT_EPSILON = sys.float_info.epsilon # Smallest representable float difference
DISPLAY_PRECISION = 2 # Decimal places for displaying weights/lengths
DEFAULT_BLADE_THICKNESS = 0.5 # mm
DEFAULT_VOXEL_SIZE = 0.0 # mm - 0 means OFF
DEFAULT_WEIGHT_TOLERANCE = 0.0 # g
DEFAULT_START_TRIM = 0.0 # mm
DEFAULT_END_TRIM = 0.0   # mm


# --- Point Cloud Generation ---
def generate_test_point_cloud(
    base_length=NOMINAL_LENGTH, base_width=NOMINAL_WIDTH, base_height=NOMINAL_HEIGHT,
    resolution=DEFAULT_RESOLUTION, noise_factor=0.03, waviness_factor = 0.05, seed=None
):
    """Generates a synthetic point cloud for a cheese loaf with imperfections."""
    if seed is not None: np.random.seed(seed)

    length = base_length * (1 + np.random.uniform(-0.05, 0.05))
    width = base_width * (1 + np.random.uniform(-0.05, 0.05))
    height = base_height * (1 + np.random.uniform(-0.05, 0.05))

    num_points_factor = 1.0 / (resolution**2)
    total_surface_area = 2 * (width * length + height * length + width * height)
    num_points = int(total_surface_area * num_points_factor * 0.1) # Adjust density factor

    points = []
    # Top/Bottom
    n_zb = max(10, int(num_points * (width * length / total_surface_area)))
    x = np.random.uniform(0, width, n_zb // 2); y = np.random.uniform(0, length, n_zb // 2)
    points.append(np.column_stack([x, y, np.full_like(x, height)])) # Top
    points.append(np.column_stack([x, y, np.zeros_like(x)]))      # Bottom
    # Front/Back
    n_xf = max(10, int(num_points * (height * length / total_surface_area)))
    y = np.random.uniform(0, length, n_xf // 2); z = np.random.uniform(0, height, n_xf // 2)
    points.append(np.column_stack([np.full_like(y, width), y, z])) # Front
    points.append(np.column_stack([np.zeros_like(y), y, z]))       # Back
    # Left/Right Ends
    n_yl = max(10, int(num_points * (width * height / total_surface_area)))
    x = np.random.uniform(0, width, n_yl // 2); z = np.random.uniform(0, height, n_yl // 2)
    points.append(np.column_stack([x, np.full_like(x, length), z])) # Right End
    points.append(np.column_stack([x, np.zeros_like(x), z]))       # Left End

    if not points: # Handle case where num_points might become zero
         return pd.DataFrame(columns=['x', 'y', 'z'])
    points = np.concatenate(points, axis=0)

    # Deformation
    x_waviness = width * waviness_factor * np.sin(points[:, 1] / length * 2 * np.pi * np.random.uniform(1.5, 3.5))
    z_waviness = height * waviness_factor * np.cos(points[:, 1] / length * 2 * np.pi * np.random.uniform(1.5, 3.5))
    side_mask = (points[:, 0] > width * 0.1) & (points[:, 0] < width * 0.9)
    if side_mask.any():
        points[side_mask, 0] += x_waviness[side_mask] * 0.5
        points[side_mask, 2] += z_waviness[side_mask]

    # Noise
    points += np.random.normal(0, resolution * 2, points.shape)

    # Clip and Center Y
    points[:, 0] = np.clip(points[:, 0], -width * 0.1, width * 1.1)
    points[:, 1] = np.clip(points[:, 1], -length * 0.1, length * 1.1)
    points[:, 2] = np.clip(points[:, 2], -height * 0.1, height * 1.1)
    if points.shape[0] > 0: points[:, 1] -= np.min(points[:, 1]) # Center Y near 0

    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return df


# --- Point Cloud Loading ---
def load_point_cloud(uploaded_file):
    """Loads point cloud data from an uploaded file (CSV, XYZ, or PCD)."""
    if uploaded_file is None: return None
    temp_path = None
    df = None
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        uploaded_file.seek(0)

        if file_ext == 'csv':
            try: # Try header first
                df_csv = pd.read_csv(uploaded_file); df_csv.columns = [c.lower() for c in df_csv.columns]
                if {'x', 'y', 'z'}.issubset(df_csv.columns): df = df_csv[['x', 'y', 'z']]
                else: raise ValueError("Missing x, y, or z")
            except Exception: # Fallback headerless
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True, names=['x', 'y', 'z'], on_bad_lines='warn', usecols=[0,1,2])
        elif file_ext == 'xyz':
            df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True, names=['x', 'y', 'z'], on_bad_lines='warn', usecols=[0,1,2])
        elif file_ext == 'pcd':
            if not _open3d_installed:
                st.error("`.pcd` requires `open3d`. Install: `pip install open3d`")
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
                temp_path = tmp.name; tmp.write(uploaded_file.getvalue())
            pcd = o3d.io.read_point_cloud(temp_path)
            if not pcd.has_points(): st.error("PCD file empty or unreadable."); return None
            points = np.asarray(pcd.points)
            if points.shape[1] != 3: st.error(f"PCD has {points.shape[1]} dims, expected 3."); return None
            df = pd.DataFrame(points, columns=['x', 'y', 'z'])
        else:
            st.error(f"Unsupported format: .{file_ext}. Use CSV, XYZ, PCD.")
            return None

        # --- Common Post-Processing ---
        if df is None: st.error("Failed to create DataFrame."); return None
        for col in ['x', 'y', 'z']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['x', 'y', 'z'])
        if df.empty: st.error("No valid numeric X, Y, Z data found."); return None

        st.success(f"Loaded {len(df)} points from {uploaded_file.name}")
        return df

    except Exception as e:
        st.error(f"Error loading '{uploaded_file.name}': {e}")
        return None
    finally: # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception as cleanup_error: st.warning(f"Could not delete temp file {temp_path}: {cleanup_error}")


# --- Core Calculation Helper: Slice Volume ---
def estimate_slice_volume_convex_hull(points_df, y_start, y_end, assume_flat_bottom=False, top_down_scan=False):
    """Estimates slice volume. Handles top-down, flat bottom, convex hull, and bbox."""
    if y_end <= y_start + FLOAT_EPSILON: return 0.0
    slice_points = points_df[(points_df['y'] >= y_start) & (points_df['y'] < y_end)]
    num_slice_points = len(slice_points)
    cross_section_area = 0.0

    if top_down_scan: # --- Top-Down Logic ---
        if num_slice_points >= 2:
            width = slice_points['x'].max() - slice_points['x'].min()
            avg_z = slice_points['z'].mean() # Assumes Z is top surface height
            height = avg_z # Assumes bottom is Z=0
            if np.isfinite(width) and np.isfinite(height) and width >= 0 and height >= 0:
                cross_section_area = width * height
    else: # --- Full 3D Logic (Hull / Bbox) ---
        if num_slice_points >= 2:
            slice_min_x, slice_max_x = slice_points['x'].min(), slice_points['x'].max()
            slice_min_z, slice_max_z = slice_points['z'].min(), slice_points['z'].max()
            # Try Convex Hull if enough points and they span an area
            if num_slice_points >= MIN_POINTS_FOR_HULL and np.ptp(slice_points[['x','z']].values, axis=0).min() > FLOAT_EPSILON * 10:
                try:
                    points_for_hull = slice_points[['x', 'z']].values
                    if assume_flat_bottom: # Add Z=0 points for hull
                        points_for_hull = np.vstack((points_for_hull, [[slice_min_x, 0.0], [slice_max_x, 0.0]]))
                    hull = ConvexHull(points_for_hull)
                    cross_section_area = hull.volume
                    if not np.isfinite(cross_section_area) or cross_section_area < 0: cross_section_area = 0.0
                except (QhullError, ValueError): pass # Hull failed, will fallback to bbox
                except Exception: pass # Other hull errors
            # Fallback to Bounding Box if Hull failed or not enough points
            if cross_section_area <= FLOAT_EPSILON:
                 effective_min_z = 0.0 if assume_flat_bottom else slice_min_z
                 width = max(0.0, slice_max_x - slice_min_x)
                 height = max(0.0, slice_max_z - effective_min_z)
                 cross_section_area = width * height
                 if not np.isfinite(cross_section_area) or cross_section_area < 0: cross_section_area = 0.0

    slice_thickness = max(0.0, y_end - y_start)
    volume = cross_section_area * slice_thickness
    return volume if (np.isfinite(volume) and volume >= 0) else 0.0


# --- Core Calculation Helper: Precise Portion Volume (Optimized) ---
def recalculate_portion_volume(volume_profile, sorted_y_starts_fwd, slice_increment_mm, portion_end_y, portion_start_y):
    """Calculates precise volume within portion boundaries using optimized slice iteration."""
    actual_portion_volume = 0.0
    if portion_start_y <= portion_end_y + FLOAT_EPSILON: return 0.0
    start_idx = max(0, np.searchsorted(sorted_y_starts_fwd, portion_end_y, side='right') - 1)
    end_idx = min(len(sorted_y_starts_fwd), np.searchsorted(sorted_y_starts_fwd, portion_start_y, side='left'))

    for i in range(start_idx, end_idx + 1):
         if i >= len(sorted_y_starts_fwd): break
         ys_calc = sorted_y_starts_fwd[i]
         vol_calc = volume_profile.get(ys_calc, 0.0)
         if vol_calc <= 0: continue
         ye_calc = sorted_y_starts_fwd[i+1] if (i + 1 < len(sorted_y_starts_fwd)) else (ys_calc + slice_increment_mm)
         overlap_start = max(ys_calc, portion_end_y)
         overlap_end = min(ye_calc, portion_start_y)
         slice_len_calc = ye_calc - ys_calc
         if overlap_end > overlap_start + FLOAT_EPSILON:
             if slice_len_calc > FLOAT_EPSILON:
                 overlap_fraction = max(0.0, min(1.0, (overlap_end - overlap_start) / slice_len_calc))
                 volume_to_add = vol_calc * overlap_fraction
                 if np.isfinite(volume_to_add) and volume_to_add >= 0:
                     actual_portion_volume += volume_to_add
                 else: return np.nan # Indicate error
    return actual_portion_volume


# --- Main Calculation Logic ---
def calculate_cut_portions_reversed(
    points_df, total_weight_g, target_portion_weight_g,
    slice_increment_mm, no_interpolation, assume_flat_bottom, top_down_scan,
    blade_thickness_mm, weight_tolerance_g,
    start_trim_mm, end_trim_mm # <-- Added Trims
    ):
    """
    Calculates cut locations (reversed), handling options including trims.
    Returns results dictionary including data for plotting.
    """
    calc_start_time = time.time()
    portions_temp = []; output_portions = []
    total_volume_mm3 = 0.0; density_g_mm3 = 0.0
    status_message = "Starting calculation..."
    progress_text = st.empty()
    volume_profile_time = 0.0; cutting_time = 0.0
    volume_profile = {}; sorted_y_starts_fwd = np.array([])

    results = { "portions": [], "total_volume": 0.0, "density": 0.0, "status": "Initialization Error", "calc_time": 0.0, "volume_profile": {}, "sorted_y_starts": np.array([]), "calc_start_y": 0.0, "calc_end_y": 0.0 } # Include calc boundaries

    # --- Input Validation ---
    if points_df is None or points_df.empty: results["status"] = "Error: No point cloud data."; return results
    if not np.all(np.isfinite(points_df[['x', 'y', 'z']].values)): results["status"] = "Error: Point cloud contains non-finite values."; return results
    if total_weight_g <= FLOAT_EPSILON: results["status"] = "Error: Total weight must be positive."; return results
    if target_portion_weight_g <= FLOAT_EPSILON: results["status"] = "Error: Target portion weight must be positive."; return results
    if slice_increment_mm <= FLOAT_EPSILON: results["status"] = "Error: Calc. Slice Thickness must be positive."; return results
    if blade_thickness_mm < 0: results["status"] = "Error: Blade Thickness cannot be negative."; return results
    if weight_tolerance_g < 0: results["status"] = "Error: Weight Tolerance cannot be negative."; return results
    if start_trim_mm < 0: results["status"] = "Error: Start Trim cannot be negative."; return results
    if end_trim_mm < 0: results["status"] = "Error: End Trim cannot be negative."; return results

    loaf_min_y, loaf_max_y = points_df['y'].min(), points_df['y'].max()
    loaf_length = loaf_max_y - loaf_min_y
    if loaf_length <= FLOAT_EPSILON: results["status"] = "Error: Invalid loaf dimensions (length near zero)."; return results

    # --- Define Calculation Range ---
    calc_start_y = loaf_min_y + start_trim_mm
    calc_end_y = loaf_max_y - end_trim_mm
    results["calc_start_y"] = calc_start_y
    results["calc_end_y"] = calc_end_y

    if calc_start_y >= calc_end_y - FLOAT_EPSILON:
        results["status"] = f"Error: Start Trim ({start_trim_mm:.1f}mm) and End Trim ({end_trim_mm:.1f}mm) overlap or exceed loaf length ({loaf_length:.1f}mm)."
        return results

    try: # Wrap main calculation steps
        # --- 1. Calculate FULL Volume Profile (for accurate density) ---
        volume_profile_start = time.time()
        full_y_steps = np.arange(loaf_min_y, loaf_max_y, slice_increment_mm) # Use full range here
        # volume_profile = {} # Init moved up
        progress_bar = st.progress(0.0)
        num_steps = len(full_y_steps) if len(full_y_steps) > 0 else 1
        effective_flat_bottom = assume_flat_bottom or top_down_scan

        for i, y_start in enumerate(full_y_steps):
            progress_text.text(f"Calculating volume profile... Slice {i+1}/{num_steps}")
            y_end = min(y_start + slice_increment_mm, loaf_max_y) # Use full max_y
            if y_start >= loaf_max_y: break
            slice_vol = estimate_slice_volume_convex_hull(points_df, y_start, y_end, effective_flat_bottom, top_down_scan)
            if slice_vol > 0: volume_profile[y_start] = slice_vol; total_volume_mm3 += slice_vol # Accumulate TOTAL volume
            progress = (i + 1) / num_steps
            if progress <= 1.0: progress_bar.progress(progress)
        volume_profile_time = time.time() - volume_profile_start
        progress_text.text(f"Volume profile calculation complete ({volume_profile_time:.2f}s).")
        if total_volume_mm3 <= FLOAT_EPSILON * 100: raise ValueError(f"Calculated TOTAL volume ({total_volume_mm3:.2e} mm³) near zero.")
        results["total_volume"] = total_volume_mm3 # Store total physical volume

        # --- 2. Calculate Density (using TOTAL weight and TOTAL volume) ---
        density_g_mm3 = total_weight_g / total_volume_mm3
        if not np.isfinite(density_g_mm3) or density_g_mm3 <= 0: raise ValueError("Calculated density is invalid.")
        results["density"] = density_g_mm3

        # --- 3. Iterate Backwards within TRIMMED Range & Determine Cuts ---
        cutting_start_time = time.time()
        current_portion_weight_accum = 0.0
        last_cut_y = calc_end_y # *** START FROM EFFECTIVE END ***
        sorted_y_starts_fwd = np.array(sorted(volume_profile.keys()))
        results["sorted_y_starts"] = sorted_y_starts_fwd # Store for plotting
        results["volume_profile"] = volume_profile # Store for plotting

        # Filter keys for backward iteration to only those relevant to the calc range
        relevant_y_indices = np.where((sorted_y_starts_fwd >= calc_start_y - slice_increment_mm) & (sorted_y_starts_fwd < calc_end_y))[0]
        if len(relevant_y_indices) == 0: raise ValueError("No volume profile slices found within the calculation range after trimming.")
        sorted_y_starts_rev_trimmed = sorted_y_starts_fwd[relevant_y_indices][::-1]

        target_weight_lower_bound = target_portion_weight_g - weight_tolerance_g

        for y_start in sorted_y_starts_rev_trimmed: # Iterate over relevant slices only
            if y_start < calc_start_y - FLOAT_EPSILON: continue # Safety check

            slice_vol = volume_profile.get(y_start, 0.0); slice_end_y = y_start + slice_increment_mm
            if slice_vol <= 0: continue
            slice_weight = slice_vol * density_g_mm3
            if not np.isfinite(slice_weight) or slice_weight < 0: slice_weight = 0.0

            if current_portion_weight_accum + slice_weight >= target_weight_lower_bound - FLOAT_EPSILON:
                remaining_volume_in_slice = 0.0; cut_y = y_start
                if not no_interpolation:
                    needed_weight_exact = target_portion_weight_g - current_portion_weight_accum
                    if slice_weight > FLOAT_EPSILON:
                        fraction = max(0.0, min(1.0, needed_weight_exact / slice_weight))
                        if slice_end_y - y_start > FLOAT_EPSILON:
                            cut_y = slice_end_y - fraction * (slice_end_y - y_start)
                            remaining_volume_in_slice = slice_vol * (1.0 - fraction)
                    else: remaining_volume_in_slice = slice_vol
                    cut_y = max(cut_y, calc_start_y) # Ensure cut stays within calc range
                # If no interp, cut_y remains y_start, remaining_vol is 0

                portion_start_y = last_cut_y
                portion_end_y = cut_y # Before kerf
                portion_length = max(0.0, portion_start_y - portion_end_y)

                actual_weight = 0.0
                if portion_length > FLOAT_EPSILON: # Avoid recalc for zero length portions
                    actual_vol = recalculate_portion_volume(volume_profile, sorted_y_starts_fwd, slice_increment_mm, portion_end_y, portion_start_y)
                    if np.isnan(actual_vol): raise ValueError("Invalid value during weight recalculation.")
                    actual_weight = actual_vol * density_g_mm3
                    if not np.isfinite(actual_weight) or actual_weight < 0: actual_weight = 0.0

                portions_temp.append({"calc_start_y": portion_start_y, "calc_end_y": portion_end_y, "length": portion_length, "weight": actual_weight})

                # Reset for next portion, applying Kerf
                last_cut_y = portion_end_y - blade_thickness_mm
                last_cut_y = max(last_cut_y, calc_start_y) # Ensure kerf doesn't push start below calc_start_y

                remaining_weight = 0.0
                if np.isfinite(remaining_volume_in_slice) and remaining_volume_in_slice > 0:
                    rem_w = remaining_volume_in_slice * density_g_mm3
                    if np.isfinite(rem_w) and rem_w > 0: remaining_weight = rem_w
                current_portion_weight_accum = remaining_weight
            else:
                if slice_weight > 0: current_portion_weight_accum += slice_weight
        cutting_time = time.time() - cutting_start_time

        # --- 4. Handle Waste Piece (Now within Trimmed Range) ---
        if last_cut_y > calc_start_y + FLOAT_EPSILON:
            first_piece_start_y = last_cut_y # Upper boundary (includes kerf from last target portion)
            first_piece_end_y = calc_start_y # Lower boundary is the effective start
            first_piece_length = max(0.0, first_piece_start_y - first_piece_end_y)

            first_piece_volume = recalculate_portion_volume(volume_profile, sorted_y_starts_fwd, slice_increment_mm, first_piece_end_y, first_piece_start_y)
            if np.isnan(first_piece_volume): raise ValueError("Invalid volume for waste piece.")
            first_piece_weight = first_piece_volume * density_g_mm3
            if not np.isfinite(first_piece_weight) or first_piece_weight < 0: first_piece_weight = 0.0
            portions_temp.insert(0, {"calc_start_y": first_piece_start_y, "calc_end_y": first_piece_end_y, "length": first_piece_length, "weight": first_piece_weight})

        # --- 5. Final Formatting & Ordering ---
        if portions_temp:
            waste_piece_raw = portions_temp[0]
            target_pieces_sorted = sorted(portions_temp[1:], key=lambda p: p.get('calc_end_y', 0.0))
            # Format Waste
            csy,cey,ln,wt = waste_piece_raw.values()
            sy,ey = (min(csy,cey),max(csy,cey)) if np.isfinite(csy) and np.isfinite(cey) else (0,0)
            output_portions.append({"portion_num": 1, "display_start_y": sy, "display_end_y": ey, "length": ln if np.isfinite(ln) else 0, "weight": wt if np.isfinite(wt) else 0, "cut_y": cey if np.isfinite(cey) else 0})
            # Format Targets
            for i, p in enumerate(target_pieces_sorted):
                csy,cey,ln,wt = p.values()
                sy,ey = (min(csy,cey),max(csy,cey)) if np.isfinite(csy) and np.isfinite(cey) else (0,0)
                output_portions.append({"portion_num": i + 2, "display_start_y": sy, "display_end_y": ey, "length": ln if np.isfinite(ln) else 0, "weight": wt if np.isfinite(wt) else 0, "cut_y": cey if np.isfinite(cey) else 0})

        if not output_portions: status_message = "Warning: No portions calculated within the specified trim range."
        else: status_message = f"Calculation complete. Found {len(output_portions)} portions within trim range (Portion 1 is waste)."

    except OverflowError: status_message = "Error: Numerical overflow during calculation."
    except ValueError as e: status_message = f"Error: {e}"
    except Exception as e: status_message = f"An unexpected error occurred: {e}"; raise
    finally:
        total_calc_time = time.time() - calc_start_time
        final_status = f"{status_message} Total Time: {total_calc_time:.2f}s (Profile: {volume_profile_time:.2f}s, Cutting: {cutting_time:.2f}s)"
        progress_text.text(final_status)
        results["portions"] = output_portions
        # results["total_volume"] already set to full loaf volume
        # results["density"] already set
        results["status"] = status_message # Base status message
        results["calc_time"] = total_calc_time
        # results["volume_profile"] and results["sorted_y_starts"] already set

    return results


# --- Visualization ---
def plot_3d_loaf(points_df, portions=None, title="Cheese Loaf Point Cloud"):
    """Creates interactive 3D plot with optional cut planes."""
    if points_df is None or points_df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points_df['x'], y=points_df['y'], z=points_df['z'], mode='markers',
        marker=dict(size=1.5, color=points_df['z'], colorscale='YlOrBr', opacity=0.7, colorbar=dict(title='Height (Z)')),
        name='Point Cloud' ))
    if portions and len(portions) > 0:
        min_x, max_x = points_df['x'].min(), points_df['x'].max(); x_rng = max(1.0, max_x - min_x)
        min_z, max_z = points_df['z'].min(), points_df['z'].max(); z_rng = max(1.0, max_z - min_z)
        plane_x = [min_x - 0.05 * x_rng, max_x + 0.05 * x_rng]
        plane_z = [min_z - 0.05 * z_rng, max_z + 0.05 * z_rng]
        cut_locations = [p['cut_y'] for p in portions[:-1]] # Cut Y is lower boundary after portion
        for i, cut_y in enumerate(cut_locations):
             if np.isfinite(cut_y):
                 fig.add_trace(go.Mesh3d(
                    x=[plane_x[0], plane_x[1], plane_x[1], plane_x[0]], y=[cut_y, cut_y, cut_y, cut_y], z=[plane_z[0], plane_z[0], plane_z[1], plane_z[1]],
                    opacity=0.4, color='red', alphahull=0, name=f"Cut after P{i+1}", showlegend= i == 0 ))
    fig.update_layout(title=title, scene=dict(xaxis_title='Width (X, mm)', yaxis_title='Length (Y, mm)', zaxis_title='Height (Z, mm)', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=40))
    return fig

# --- 2D Plotting Functions ---
def plot_area_profile(volume_profile, sorted_y_starts, slice_increment_mm, calc_start_y=None, calc_end_y=None):
    """Plots area profile, optionally highlighting calculation range."""
    if not volume_profile or len(sorted_y_starts) == 0:
        return go.Figure(layout=dict(title="Area Profile (No Data)", height=300))
    y_values = sorted_y_starts
    area_values = [(volume_profile.get(y, 0) / slice_increment_mm if slice_increment_mm > FLOAT_EPSILON else 0) for y in y_values]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_values, y=area_values, mode='lines', name='Est. Area'))
    # Add shaded regions for trims
    max_area = max(area_values) if area_values else 1
    plot_min_y = y_values[0] if len(y_values) > 0 else 0
    plot_max_y = y_values[-1] + slice_increment_mm if len(y_values) > 0 else 0
    if calc_start_y is not None and calc_start_y > plot_min_y:
        fig.add_vrect(x0=plot_min_y, x1=calc_start_y, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="Start Trim")
    if calc_end_y is not None and calc_end_y < plot_max_y:
         fig.add_vrect(x0=calc_end_y, x1=plot_max_y, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="End Trim")
    fig.update_layout(title="Area Profile (Shaded=Trimmed)", xaxis_title="Length (Y, mm)", yaxis_title="Area (mm²)", margin=dict(l=20,r=20,t=40,b=20), height=300)
    return fig

def plot_cumulative_weight(volume_profile, sorted_y_starts, density_g_mm3, portions, target_weight, tolerance, calc_start_y=None, calc_end_y=None):
    """Plots cumulative weight, cuts, and optional trim range."""
    if not volume_profile or len(sorted_y_starts) == 0 or density_g_mm3 <= 0:
        return go.Figure(layout=dict(title="Cumulative Weight (No Data)", height=350))

    y_coords = []; cumulative_weights = []; current_weight = 0.0
    if sorted_y_starts.size > 0:
        y_coords.append(sorted_y_starts[0]); cumulative_weights.append(0.0) # Start at min Y
        for i, y_start in enumerate(sorted_y_starts):
            slice_vol = volume_profile.get(y_start, 0.0)
            current_weight += (slice_vol * density_g_mm3)
            # Use next y_start for end point for better accuracy if available
            y_end = sorted_y_starts[i+1] if (i + 1 < len(sorted_y_starts)) else (y_start + (sorted_y_starts[1]-sorted_y_starts[0] if len(sorted_y_starts)>1 else 1.0))
            y_coords.append(y_end); cumulative_weights.append(current_weight)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_coords, y=cumulative_weights, mode='lines', name='Cumulative Weight'))
    # Add target/tolerance lines
    fig.add_hline(y=target_weight, line_dash="dash", line_color="red", name="Target")
    if tolerance > 0:
        fig.add_hline(y=target_weight + tolerance, line_dash="dot", line_color="orange", name="+ Tolerance")
        fig.add_hline(y=target_weight - tolerance, line_dash="dot", line_color="orange", name="- Tolerance")
    max_weight_plot = max(cumulative_weights) if cumulative_weights else target_weight * 1.1
    # Add vertical lines for cuts
    for i, p in enumerate(portions):
        cut_y = p.get('cut_y', np.nan)
        # Only draw cut lines if they represent an actual portion end (not the final end)
        if np.isfinite(cut_y) and i < len(portions) - 1 and p.get('length', 0) > FLOAT_EPSILON:
             fig.add_vline(x=cut_y, line_dash="solid", line_color="grey", name=f"Cut {i+1}" if i<3 else None, showlegend=(i==0))
    # Add shaded regions for trims
    plot_min_y = y_coords[0] if y_coords else 0
    plot_max_y = y_coords[-1] if y_coords else 0
    if calc_start_y is not None and calc_start_y > plot_min_y:
        fig.add_vrect(x0=plot_min_y, x1=calc_start_y, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="Start Trim")
    if calc_end_y is not None and calc_end_y < plot_max_y:
         fig.add_vrect(x0=calc_end_y, x1=plot_max_y, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="End Trim")
    fig.update_layout(title="Cumulative Weight (Shaded=Trimmed)", xaxis_title="Length (Y, mm)", yaxis_title="Cumulative Weight (g)", yaxis_range=[0, max_weight_plot * 1.05], margin=dict(l=20,r=20,t=40,b=20), height=350)
    return fig

# -----------------------------------------------------------------------------
# Streamlit Application UI
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Cheese Portion Calc V2.6")

# --- Initialize Session State ---
default_state = {
    'point_cloud_data': None, 'data_origin': None, 'last_file_id': None,
    'calc_results': None # Store all calculation results here
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Sidebar ---
with st.sidebar:
    st.header("🧀 Loaf Parameters")
    data_source = st.radio("Point Cloud Source:", ("Generate Test Data", "Upload File"), index=0, key="data_source")
    uploaded_file = None
    file_types = ['csv', 'xyz']
    if _open3d_installed: file_types.append('pcd')
    if st.session_state.data_source == "Upload File":
        uploaded_file = st.file_uploader(f"Choose a .{', .'.join(file_types)} file", type=file_types, key="uploader")
        if 'pcd' in file_types and not _open3d_installed:
             st.warning("Install `open3d` to load `.pcd` files.")
        elif 'pcd' in file_types:
             st.caption("`.pcd` requires `open3d` library.")
    else: st.markdown("_(Using generated test data)_")

    st.number_input("Total Loaf Weight (g)", min_value=0.01, value=float(DEFAULT_TOTAL_WEIGHT), step=50.0, format="%.2f", key="total_weight")
    st.number_input("Target Portion Weight (g)", min_value=0.01, value=DEFAULT_TARGET_WEIGHT, step=10.0, format="%.2f", key="target_weight")
    st.number_input("Weight Tolerance (+/- g)", min_value=0.0, value=DEFAULT_WEIGHT_TOLERANCE, step=0.5, format="%.1f", key="weight_tolerance", help="Allowed deviation from target weight (+/- grams).")

    st.divider()
    st.subheader("Scan & Calculation Settings")
    st.number_input("Start Trim (mm)", min_value=0.0, value=DEFAULT_START_TRIM, step=1.0, format="%.1f", key="start_trim", help="Length to discard from the start (min Y) before portioning. Useful for removing noise or inconsistent leading edge data.")
    st.number_input("End Trim (mm)", min_value=0.0, value=DEFAULT_END_TRIM, step=1.0, format="%.1f", key="end_trim", help="Length to discard from the end (max Y) before portioning. Useful for removing noise or inconsistent falling edge data.")
    st.number_input("Scanner Resolution (mm)", min_value=0.01, value=DEFAULT_RESOLUTION, step=0.1, format="%.2f", help="Informational. Used for test data generation.", key="resolution")
    st.slider("Calc. Slice Thickness (mm)", min_value=0.1, max_value=10.0, value=DEFAULT_SLICE_THICKNESS, step=0.1, format="%.1f", key="slice_thickness", help="Smaller: +Accuracy / -Speed.")
    st.number_input("Blade Thickness / Kerf (mm)", min_value=0.0, value=DEFAULT_BLADE_THICKNESS, step=0.1, format="%.1f", key="blade_thickness", help="Material width removed by each cut.")
    st.number_input("Downsample Voxel Size (mm)", min_value=0.0, value=DEFAULT_VOXEL_SIZE, step=0.5, format="%.1f", key="voxel_size", help="0 = OFF. Reduces points to speed up calculation. Requires 'open3d'.")
    if st.session_state.voxel_size > 0 and not _open3d_installed:
        st.warning("Downsampling requires `open3d`, but it's not installed. Setting ignored.")

    st.divider()
    st.subheader("Scan Simulation & Method")
    st.toggle("Cut > Target (No Interpolation)", value=False, key="no_interp", help="ON: Cut after target met (overweight). OFF: Interpolate cut.")
    st.toggle("Top-Down Scan Only", value=False, key="top_down_scan", help="ON: Assumes only top Z known (vertical sides, Z=0 bottom). Overrides 'Flat Bottom'. One scan only option.")
    flat_bottom_disabled = st.session_state.get('top_down_scan', False)
    st.toggle("Top & Side Scan Only", value=st.session_state.get('flat_bottom', False), key="flat_bottom", help="ON: Assumes Z=0 bottom. Ignored if 'Top-Down Scan' is ON. Two scan only option.", disabled=flat_bottom_disabled)

    st.divider()
    calc_button = st.button("Calculate Portions", type="primary")
    st.info("Instructions:\n1. Set source & params.\n2. **Portion 1** is the *first* physical piece (min Y) = waste.\n3. Click 'Calculate Portions'.")

# --- Main Area ---
st.title("🔪 Cheese Loaf Portion Calculator (V2.6)")
st.markdown("Calculates cut locations using volume estimation. Portion 1 is the first physical piece (waste).")

# --- Data Loading / Generation Trigger ---
refresh_data = False
if st.session_state.data_source == "Generate Test Data":
    if st.session_state.data_origin != "Generated" or st.session_state.point_cloud_data is None: refresh_data = True
elif st.session_state.data_source == "Upload File" and uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if current_file_id != st.session_state.last_file_id: refresh_data = True

if calc_button or refresh_data:
    st.session_state.calc_results = None # Clear previous results on new data/calc trigger

    # Load or Generate base points_df
    temp_points_df = None
    if st.session_state.data_source == "Generate Test Data":
         with st.spinner("Generating test point cloud..."):
            temp_points_df = generate_test_point_cloud(resolution=st.session_state.resolution, seed=42)
            st.session_state.data_origin = "Generated"
            if temp_points_df is not None: st.info(f"Generated {len(temp_points_df)} points.")
            else: st.error("Failed to generate test data.")
    elif st.session_state.data_source == "Upload File" and uploaded_file is not None:
        with st.spinner(f"Loading point cloud from {uploaded_file.name}..."):
            temp_points_df = load_point_cloud(uploaded_file)
            if temp_points_df is not None:
                st.session_state.data_origin = f"Uploaded: {uploaded_file.name}"
                st.session_state.last_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            else: st.session_state.data_origin = "Upload Failed"; st.session_state.last_file_id = None

    # --- Apply Downsampling ---
    if temp_points_df is not None and not temp_points_df.empty and st.session_state.voxel_size > 0:
        if _open3d_installed:
            with st.spinner(f"Downsampling point cloud (Voxel Size: {st.session_state.voxel_size:.1f} mm)..."):
                pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(temp_points_df[['x', 'y', 'z']].values)
                downsampled_pcd = pcd.voxel_down_sample(voxel_size=st.session_state.voxel_size)
                downsampled_points = np.asarray(downsampled_pcd.points)
                if downsampled_points.shape[0] > 0:
                    st.session_state.point_cloud_data = pd.DataFrame(downsampled_points, columns=['x', 'y', 'z'])
                    st.info(f"Downsampled to {len(st.session_state.point_cloud_data)} points.")
                else:
                    st.warning("Downsampling resulted in zero points. Using original data.")
                    st.session_state.point_cloud_data = temp_points_df
        else:
            st.warning("Voxel downsampling ignored: `open3d` not installed.")
            st.session_state.point_cloud_data = temp_points_df
    else:
         st.session_state.point_cloud_data = temp_points_df # Assign loaded/generated data

# --- Display Point Cloud Info and 3D Plot ---
st.subheader("Point Cloud Data")
points_df = st.session_state.point_cloud_data
if points_df is not None and not points_df.empty:
    st.markdown(f"**Source:** {st.session_state.data_origin}")
    col1, col2, col3 = st.columns(3)
    try: # Display basic dimensions
        if np.all(np.isfinite(points_df[['x', 'y', 'z']])):
            min_dims=points_df.min(); max_dims=points_df.max(); dims=max_dims-min_dims
            col1.metric("Points Loaded", f"{len(points_df):,}")
            col2.metric("Loaf Length (Y)", f"{dims['y']:.1f} mm")
            col3.metric("Est. Max Width (X)", f"{dims['x']:.1f} mm")
        else: col1.metric("Points Loaded", f"{len(points_df):,}"); col2.warning("Non-finite values.")
    except Exception as e: st.warning(f"Could not get dimensions: {e}")

    with st.spinner("Generating 3D plot..."):
        plot_title = "Point Cloud with Calculated Cuts" if st.session_state.calc_results and st.session_state.calc_results.get("portions") else "Loaded Point Cloud"
        plot_portions = st.session_state.calc_results.get("portions") if st.session_state.calc_results else None
        fig3d = plot_3d_loaf(points_df, portions=plot_portions, title=plot_title)
        st.plotly_chart(fig3d, use_container_width=True)
else: st.warning("Please generate or upload point cloud data.")

# --- Calculation Trigger ---
if calc_button and points_df is not None and not points_df.empty:
    if np.all(np.isfinite(points_df[['x', 'y', 'z']])):
        with st.spinner("Calculating cuts..."):
            effective_flat_bottom = st.session_state.flat_bottom or st.session_state.top_down_scan
            # Store all results in a single dictionary in session state
            st.session_state.calc_results = calculate_cut_portions_reversed(
                points_df, st.session_state.total_weight, st.session_state.target_weight,
                st.session_state.slice_thickness, st.session_state.no_interp,
                effective_flat_bottom, st.session_state.top_down_scan,
                st.session_state.blade_thickness, st.session_state.weight_tolerance,
                st.session_state.start_trim, st.session_state.end_trim # Pass trims
            )
            st.rerun() # Force rerun to display results and updated plots
    else:
        st.error("Cannot calculate - point cloud contains non-finite values.")
        st.session_state.calc_results = {"status": "Error: Non-finite values in point cloud."}

# --- Display Results ---
st.subheader("Portioning Results")
calc_results = st.session_state.calc_results
if calc_results:
    status_message = calc_results.get("status", "Status unknown")
    portions = calc_results.get("portions", [])

    if "Error:" in status_message: st.error(status_message)
    elif "Warning:" in status_message: st.warning(status_message)
    else: st.info(status_message) # Includes timing

    if portions: # Display table and metrics only if portions exist
        results_data = []; total_calculated_weight = 0
        for p in portions:
            results_data.append({ "Portion #": p.get('portion_num'), "Start Y (mm)": f"{p.get('display_start_y'):.{DISPLAY_PRECISION}f}", "End Y (mm)": f"{p.get('display_end_y'):.{DISPLAY_PRECISION}f}", "Length (mm)": f"{p.get('length'):.{DISPLAY_PRECISION}f}", "Est. Weight (g)": f"{p.get('weight'):.{DISPLAY_PRECISION}f}" })
            total_calculated_weight += p.get('weight', 0.0)
        results_df = pd.DataFrame(results_data)

        col_table, col_metrics = st.columns([3, 1.5])
        with col_table:
             st.markdown("**Calculated Portions (Portion 1 is waste):**"); st.dataframe(results_df, use_container_width=True, hide_index=True)
             # --- Add Export Button ---
             csv = results_df.to_csv(index=False).encode('utf-8')
             st.download_button("Download Results CSV", csv, "cheese_portions.csv", "text/csv", key='download-csv')
             # --- End Export Button ---
        with col_metrics:
             st.metric("Target Weight", f"{st.session_state.target_weight:.{DISPLAY_PRECISION}f} g")
             st.metric("Tolerance", f"± {st.session_state.weight_tolerance:.{DISPLAY_PRECISION}f} g")
             st.metric("Portion 1 Weight", f"{portions[0].get('weight', 0.0):.{DISPLAY_PRECISION}f} g", help="Waste piece.")
             weight_delta = total_calculated_weight - st.session_state.total_weight
             delta_color = "off" if abs(weight_delta) < 0.1 else ("inverse" if weight_delta < 0 else "normal")
             st.metric("Total Calc. Portion Weight", f"{total_calculated_weight:.{DISPLAY_PRECISION}f} g", delta=f"{weight_delta:.{DISPLAY_PRECISION}f} g vs Input", delta_color=delta_color, help="Sum of weights of calculated portions (excludes trims, kerf loss).")
             st.caption(f"Density: {calc_results.get('density',0)*1e6:,.2f} kg/m³ | Vol: {calc_results.get('total_volume',0)/1e3:,.1f} cm³")
             # Rounding Note
             if any(abs(p.get('weight', 0) - st.session_state.target_weight) < (10**(-DISPLAY_PRECISION))/2 for p in portions[1:]):
                  note = f"Note: Weights rounded."; note += " Exact target hit w/o interpolation is unusual." if st.session_state.no_interp else ""
                  st.caption(note)

        # --- Display 2D Plots ---
        st.divider()
        st.subheader("Calculation Profiles")
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            with st.spinner("Generating area plot..."):
                fig_area = plot_area_profile(calc_results.get("volume_profile",{}), calc_results.get("sorted_y_starts",np.array([])), st.session_state.slice_thickness, calc_results.get("calc_start_y"), calc_results.get("calc_end_y"))
                st.plotly_chart(fig_area, use_container_width=True)
        with col_plot2:
            with st.spinner("Generating weight plot..."):
                fig_weight = plot_cumulative_weight(calc_results.get("volume_profile",{}), calc_results.get("sorted_y_starts",np.array([])), calc_results.get("density",0), portions, st.session_state.target_weight, st.session_state.weight_tolerance, calc_results.get("calc_start_y"), calc_results.get("calc_end_y"))
                st.plotly_chart(fig_weight, use_container_width=True)

    elif not portions and "Error:" not in status_message and "Warning:" not in status_message:
         st.warning("Calculation ran but resulted in zero portions.")

elif not calc_button: # Initial state
     st.info("Configure parameters and click 'Calculate Portions'.")

# --- End of Script ---
