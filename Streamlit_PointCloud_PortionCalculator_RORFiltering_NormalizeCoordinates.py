# -----------------------------------------------------------------------------
# Point Cloud Portion Calculator (Streamlit App)
# Created - Shaun Harris
# Version: 1.4.12 (Features: File Uploads (CSV/XYZ/PCD/PYL/XLSX/XLS), Test Data Gen,
#                 Volume Modes (Hull, Flat Bottom, Top-Down), Reverse Calc,
#                 Waste First, Interpolation Toggle, Kerf, Tolerance, Trims,
#                 Voxel Downsampling, Optional Auto-Downsampling (Random) with User Threshold,
#                 Radius Outlier Removal (ROR), Plotting (3D + 2D Profiles),
#                 Optimized XYZ loading, UI Fixes, Improved loading feedback,
#                 Decoupled data refresh, Optional Y-Normalization before calculation,
#                 Direct Density Input, Open3D Fly Around, Open3D Static Cuts View,
#                 Reset Button, Help Section, NumPy Optimized Volume Profiling)
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
import openpyxl # for .xlsx/.xls support

# --- Optional: Import open3d with error handling ---
_open3d_installed = False
try:
    import open3d as o3d
    _open3d_installed = True
except ImportError:
    pass
# --- End Optional ---

# --- Constants and Configuration ---
NOMINAL_HEIGHT = 90.0; NOMINAL_WIDTH = 93.33; NOMINAL_LENGTH = 360.0
DEFAULT_TARGET_WEIGHT = 250.0
DEFAULT_TOTAL_WEIGHT = (NOMINAL_WIDTH * NOMINAL_HEIGHT * NOMINAL_LENGTH) * 1.05e-9 * 1050 * 1000
DEFAULT_RESOLUTION = 0.3; DEFAULT_SLICE_THICKNESS = 0.5
MIN_POINTS_FOR_HULL = 3; FLOAT_EPSILON = sys.float_info.epsilon
DISPLAY_PRECISION = 2; DEFAULT_BLADE_THICKNESS = 0.0
DEFAULT_VOXEL_SIZE = 0.0; DEFAULT_WEIGHT_TOLERANCE = 0.0
DEFAULT_START_TRIM = 0.0; DEFAULT_END_TRIM = 0.0
DEFAULT_DIRECT_DENSITY_G_CM3 = 1.05  # replace with your default
DEFAULT_AUTO_DOWNSAMPLE_THRESHOLD = 350000


# --- Point Cloud Generation ---
def generate_test_point_cloud(
    base_length=NOMINAL_LENGTH, base_width=NOMINAL_WIDTH, base_height=NOMINAL_HEIGHT,
    resolution=DEFAULT_RESOLUTION, noise_factor=0.03, waviness_factor = 0.05, seed=None
):
    if seed is not None: np.random.seed(seed)
    length = base_length*(1+np.random.uniform(-0.05,0.05)); width=base_width*(1+np.random.uniform(-0.05,0.05)); height=base_height*(1+np.random.uniform(-0.05,0.05))
    num_pts_factor=1.0/(resolution**2); total_surf_area=2*(width*length+height*length+width*height)
    num_pts=int(total_surf_area*num_pts_factor*0.1); pts_list=[]
    n_zb=max(10,int(num_pts*(width*length/total_surf_area))); x=np.random.uniform(0,width,n_zb//2); y=np.random.uniform(0,length,n_zb//2)
    pts_list.append(np.column_stack([x,y,np.full_like(x,height)])); pts_list.append(np.column_stack([x,y,np.zeros_like(x)]))
    n_xf=max(10,int(num_pts*(height*length/total_surf_area))); y=np.random.uniform(0,length,n_xf//2); z=np.random.uniform(0,height,n_xf//2)
    pts_list.append(np.column_stack([np.full_like(y,width),y,z])); pts_list.append(np.column_stack([np.zeros_like(y),y,z]))
    n_yl=max(10,int(num_pts*(width*height/total_surf_area))); x=np.random.uniform(0,width,n_yl//2); z=np.random.uniform(0,height,n_yl//2)
    pts_list.append(np.column_stack([x,np.full_like(x,length),z])); pts_list.append(np.column_stack([x,np.zeros_like(x),z]))
    if not pts_list: return pd.DataFrame(columns=['x','y','z'])
    pts_arr=np.concatenate(pts_list,axis=0)
    if pts_arr.shape[0]>0:
        safe_len=length if length>FLOAT_EPSILON else 1.0
        x_wave=width*waviness_factor*np.sin(pts_arr[:,1]/safe_len*2*np.pi*np.random.uniform(1.5,3.5))
        z_wave=height*waviness_factor*np.cos(pts_arr[:,1]/safe_len*2*np.pi*np.random.uniform(1.5,3.5))
        s_mask=(pts_arr[:,0]>width*0.1)&(pts_arr[:,0]<width*0.9)
        if np.any(s_mask): pts_arr[s_mask,0]+=x_wave[s_mask]*0.5; pts_arr[s_mask,2]+=z_wave[s_mask]
        pts_arr+=np.random.normal(0,resolution*2,pts_arr.shape)
        pts_arr[:,0]=np.clip(pts_arr[:,0],-width*0.1,width*1.1); pts_arr[:,1]=np.clip(pts_arr[:,1],-length*0.1,length*1.1); pts_arr[:,2]=np.clip(pts_arr[:,2],-height*0.1,height*1.1)
        if pts_arr.shape[0]>0: pts_arr[:,1]-=np.min(pts_arr[:,1])
    else: return pd.DataFrame(columns=['x','y','z'])
    return pd.DataFrame(pts_arr,columns=['x','y','z'])

# --- Point Cloud Loading ---
def load_point_cloud(uploaded_file):
    if uploaded_file is None: return None
    temp_path = None; df = None
    try:
        file_name_lower = uploaded_file.name.lower(); file_ext = ""
        if "." in file_name_lower: file_ext = file_name_lower.split('.')[-1]
        uploaded_file.seek(0)
        o3d_xyz_success_flag_local = False
        if file_ext == 'csv':
            try:
                df_csv = pd.read_csv(uploaded_file); df_csv.columns = [c.lower() for c in df_csv.columns]
                if {'x','y','z'}.issubset(df_csv.columns): df = df_csv[['x','y','z']]
                else: raise ValueError("Missing x,y,z cols")
            except Exception:
                uploaded_file.seek(0); df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True, names=['x','y','z'], on_bad_lines='warn', usecols=[0,1,2])
        elif file_ext == 'xyz':
            if _open3d_installed:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xyz") as tmp: temp_path = tmp.name; tmp.write(uploaded_file.getvalue())
                    pcd_xyz_load = o3d.io.read_point_cloud(temp_path)
                    if not pcd_xyz_load.has_points():
                        st.warning(f"O3D read {uploaded_file.name} but empty. Pandas fallback.")
                        uploaded_file.seek(0); df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True, names=['x','y','z'], on_bad_lines='warn', usecols=[0,1,2])
                    else:
                        points = np.asarray(pcd_xyz_load.points)
                        if points.shape[1]!=3: st.error(f"O3D read {uploaded_file.name} but {points.shape[1]} dims (exp 3)."); return None
                        df = pd.DataFrame(points, columns=['x','y','z']); o3d_xyz_success_flag_local = True
                except Exception as e_o3d_xyz:
                    st.warning(f"O3D failed for XYZ '{uploaded_file.name}': {e_o3d_xyz}. Pandas fallback..."); uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True, names=['x','y','z'], on_bad_lines='warn', usecols=[0,1,2])
            else:
                st.info("O3D not installed. Using Pandas for .xyz (slow for large files)."); df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True, names=['x','y','z'], on_bad_lines='warn', usecols=[0,1,2])
        elif file_ext in ['pcd', 'ply']:
            if not _open3d_installed: st.error(f".{file_ext} needs o3d."); return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp: temp_path = tmp.name; tmp.write(uploaded_file.getvalue())
            pcd_load = o3d.io.read_point_cloud(temp_path)
            if not pcd_load.has_points(): st.error(f"{file_ext.upper()} empty/unreadable."); return None
            points = np.asarray(pcd_load.points)
            if points.shape[1]!=3: st.error(f"{file_ext.upper()} has {points.shape[1]} dims (exp 3)."); return None
            df = pd.DataFrame(points, columns=['x','y','z'])
        elif file_ext in ['xlsx', 'xls']:
            try:
                df_excel = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl' if file_ext=='xlsx' else None)
                df_excel.columns = [str(c).lower() for c in df_excel.columns]
                if {'x','y','z'}.issubset(df_excel.columns): df = df_excel[['x','y','z']].copy()
                else:
                    if df_excel.shape[1]>=3: st.warning("Cols 'x,y,z' not found. Assuming first 3."); df=df_excel.iloc[:,0:3].copy(); df.columns=['x','y','z']
                    else: raise ValueError("Not enough cols in Excel.")
            except ImportError: st.error("Excel needs openpyxl."); return None
            except Exception as ex_err: st.error(f"Excel error: {ex_err}"); return None
        else:
            sup_fmts = "CSV, XYZ" + (", PCD, PLY" if _open3d_installed else "") + ", XLSX, XLS"; st.error(f"Unsupported: .{file_ext}. Use {sup_fmts}."); return None
        if df is None: st.error("Failed to create DataFrame from file."); return None
        for col in ['x','y','z']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['x','y','z'])
        if df.empty: st.error("No valid numeric X,Y,Z data found in file."); return None
        if o3d_xyz_success_flag_local: st.toast(f"Loaded {len(df)} pts from {uploaded_file.name} via Open3D.", icon="🚀")
        elif file_ext != 'xyz': st.toast(f"Loaded {len(df)} pts from {uploaded_file.name}.", icon="✅")
        elif df is not None : st.toast(f"Loaded {len(df)} pts from {uploaded_file.name} using Pandas.", icon="✅")
        return df
    except Exception as e: st.error(f"Load error '{uploaded_file.name}': {e}"); return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception as cl_err: st.warning(f"Temp file cleanup error: {cl_err}")

def estimate_ror_radius_util_o3d(o3d_pcd, k_neighbors, mult):
    if o3d_pcd is None or not o3d_pcd.has_points(): return None, "Cloud empty for ROR est."
    n_pts = len(o3d_pcd.points)
    if n_pts < k_neighbors + 1: return None, f"Not enough pts ({n_pts}) for k={k_neighbors}."
    try:
        tree = o3d.geometry.KDTreeFlann(o3d_pcd); dists = []
        samples = min(1000, n_pts); indices = np.random.choice(n_pts, size=samples, replace=False)
        for i in indices:
            [k_found, idx, _] = tree.search_knn_vector_3d(o3d_pcd.points[i], k_neighbors + 1)
            if k_found >= k_neighbors + 1: dists.append(np.linalg.norm(o3d_pcd.points[i] - o3d_pcd.points[idx[k_neighbors]]))
        if not dists: return None, "No k-th neighbors found. Cloud sparse or k too high."
        avg_dist = np.mean(dists); est_rad = avg_dist * mult
        return est_rad, f"Avg k-NN dist ({samples} samples, k={k_neighbors}): {avg_dist:.4f}mm. Suggested Radius (x{mult:.2f}): {est_rad:.4f}mm"
    except Exception as e: return None, f"ROR radius est. error: {e}"

# --- MODIFIED FOR NUMPY OPTIMIZATION ---
def estimate_slice_volume_convex_hull_numpy(
    y_coords_np, x_coords_np, z_coords_np,
    y_s, y_e,
    flat_bottom, top_down
):
    if y_e <= y_s + FLOAT_EPSILON:
        return 0.0

    start_idx = np.searchsorted(y_coords_np, y_s, side='left')
    end_idx = np.searchsorted(y_coords_np, y_e, side='left')

    if start_idx >= end_idx:
        return 0.0

    slice_x_np = x_coords_np[start_idx:end_idx]
    slice_z_np = z_coords_np[start_idx:end_idx]

    n_sl_pts = len(slice_x_np)
    area = 0.0

    if top_down:
        if n_sl_pts >= 2:
            w = slice_x_np.max() - slice_x_np.min()
            h_z = np.mean(slice_z_np)
            if np.isfinite(w) and np.isfinite(h_z) and w >= 0 and h_z >= 0:
                area = w * h_z
    else:
        if n_sl_pts >= 2:
            sl_min_x, sl_max_x = slice_x_np.min(), slice_x_np.max()
            sl_min_z_actual, sl_max_z_actual = slice_z_np.min(), slice_z_np.max()
            eff_min_z_for_area_calc = 0.0 if flat_bottom else sl_min_z_actual
            width_for_area = sl_max_x - sl_min_x
            height_for_area = sl_max_z_actual - eff_min_z_for_area_calc

            if n_sl_pts >= MIN_POINTS_FOR_HULL and \
               width_for_area > FLOAT_EPSILON * 10 and \
               height_for_area > FLOAT_EPSILON * 10:
                try:
                    hull_pts_np = np.column_stack((slice_x_np, slice_z_np))
                    if flat_bottom:
                        hull_pts_np = np.vstack((hull_pts_np,
                                                 [[sl_min_x, 0.0], [sl_max_x, 0.0]]))
                    hull = ConvexHull(hull_pts_np)
                    area = hull.volume
                    if not (np.isfinite(area) and area >= 0): area = 0.0
                except (QhullError, ValueError, Exception): area = 0.0
            if area <= FLOAT_EPSILON:
                area = max(0.0, width_for_area) * max(0.0, height_for_area)
            if not (np.isfinite(area) and area >= 0): area = 0.0

    slice_length = max(0.0, y_e - y_s)
    volume = area * slice_length
    return volume if (np.isfinite(volume) and volume >=0) else 0.0

def recalculate_portion_volume(vol_prof, sorted_y_s, slice_inc, p_min_y, p_max_y):
    act_vol = 0.0
    if p_max_y <= p_min_y + FLOAT_EPSILON: return 0.0
    s_idx = np.searchsorted(sorted_y_s, p_min_y - slice_inc, side='left')
    e_idx = np.searchsorted(sorted_y_s, p_max_y, side='right')
    for i in range(s_idx, e_idx):
        if i >= len(sorted_y_s): break
        ys_c = sorted_y_s[i]; vol_c = vol_prof.get(ys_c, 0.0)
        if vol_c <=0: continue
        ye_c = sorted_y_s[i+1] if (i+1 < len(sorted_y_s)) else (ys_c + slice_inc)
        ov_min = max(ys_c, p_min_y); ov_max = min(ye_c, p_max_y)
        sl_len_c = ye_c - ys_c
        if ov_max > ov_min + FLOAT_EPSILON:
            if sl_len_c > FLOAT_EPSILON:
                ov_frac = max(0.0, min(1.0, (ov_max - ov_min) / sl_len_c))
                v_add = vol_c * ov_frac
                if np.isfinite(v_add) and v_add >=0: act_vol += v_add
                else: return np.nan
    return act_vol

# --- MODIFIED FOR NUMPY OPTIMIZATION ---
def calculate_cut_portions_reversed(
    points_df, total_w, target_w, slice_inc, no_interp, flat_bottom, top_down,
    blade_thick, w_tol, start_trim, end_trim, direct_density_g_mm3=None ):
    calc_t_start = time.time(); tmp_portions=[]; out_portions=[]
    tot_vol=0.0; dens=0.0; stat_msg="Starting..."
    prog_txt_area = st.empty(); prog_txt_area.text(stat_msg)
    vp_time=0.0; cut_time=0.0; vp_prof={}; sorted_y_s_arr=np.array([])
    res = {"portions":[], "total_volume":0.0, "density":0.0, "status":"Init Err", "calc_time":0.0,
           "volume_profile":{}, "sorted_y_starts":np.array([]), "calc_start_y":0.0, "calc_end_y":0.0,
           "y_offset_for_plot": 0.0, "density_source_message": ""}
    if points_df is None or points_df.empty: res["status"]="Err: No cloud data."; return res
    if not np.all(np.isfinite(points_df[['x','y','z']].values)): res["status"]="Err: Cloud has non-finite values."; return res
    loaf_min_y_orig_calc, loaf_max_y_orig_calc = points_df['y'].min(), points_df['y'].max()
    if loaf_max_y_orig_calc - loaf_min_y_orig_calc <= FLOAT_EPSILON: res["status"]="Err: Loaf length (for calc) near zero."; return res
    calc_sy_eff = loaf_min_y_orig_calc + start_trim; calc_ey_eff = loaf_max_y_orig_calc - end_trim
    res["calc_start_y"] = calc_sy_eff; res["calc_end_y"] = calc_ey_eff
    if calc_sy_eff >= calc_ey_eff - FLOAT_EPSILON: res["status"]=f"Err: Trims overlap (Eff. length: {calc_ey_eff-calc_sy_eff:.2f}mm)."; return res

    try:
        _vp_start_t = time.time()
        prog_txt_area.text("Preparing point cloud (sorting by Y)...")
        points_df_sorted = points_df.sort_values(by='y', kind='mergesort', ignore_index=True)
        y_coords_np = points_df_sorted['y'].to_numpy(dtype=np.float64, copy=False) 
        x_coords_np = points_df_sorted['x'].to_numpy(dtype=np.float64, copy=False)
        z_coords_np = points_df_sorted['z'].to_numpy(dtype=np.float64, copy=False)
        prog_txt_area.text("Point cloud prepared for profiling.")

        full_y_steps = np.arange(loaf_min_y_orig_calc, loaf_max_y_orig_calc, slice_inc)
        _pb_area = st.empty(); _pb = _pb_area.progress(0.0)
        n_steps = len(full_y_steps) if len(full_y_steps)>0 else 1
        eff_fb_vp = flat_bottom or top_down

        for i, y_s_loop in enumerate(full_y_steps):
            prog_txt_area.text(f"Vol profile... Slice {i+1}/{n_steps} (Y={y_s_loop:.1f}mm)")
            y_e_loop = min(y_s_loop + slice_inc, loaf_max_y_orig_calc)
            if y_s_loop >= loaf_max_y_orig_calc - FLOAT_EPSILON: break

            sl_vol = estimate_slice_volume_convex_hull_numpy( # CALLING THE NUMPY VERSION
                y_coords_np, x_coords_np, z_coords_np,
                y_s_loop, y_e_loop,
                eff_fb_vp, top_down
            )
            if sl_vol > 0: vp_prof[y_s_loop] = sl_vol; tot_vol += sl_vol
            _prog_val = (i+1)/n_steps
            if _prog_val <=1.0: _pb.progress(_prog_val)
       
        vp_time = time.time() - _vp_start_t
        prog_txt_area.text(f"Vol profile done ({vp_time:.2f}s). Est. Total Vol: {tot_vol/1000:.1f} cm³.")
        if direct_density_g_mm3 is not None and direct_density_g_mm3 > FLOAT_EPSILON:
            dens = direct_density_g_mm3
            res["density_source_message"] = f"Using direct density: {dens * 1000:.3f} g/cm³."
        else:
            if tot_vol <= FLOAT_EPSILON*100: raise ValueError(f"Est. total volume ({tot_vol:.2e}mm³) is near zero.")
            if total_w <= FLOAT_EPSILON: raise ValueError("Total weight must be >0 to calculate density.")
            dens = total_w / tot_vol
            res["density_source_message"] = f"Calculated density: {dens * 1000:.3f} g/cm³ (from Total Wt: {total_w:.2f}g / Est. Vol: {tot_vol/1000:.1f}cm³). Compute Time: {vp_time:.2f}s"
        if not (np.isfinite(dens) and dens > 0): raise ValueError(f"Density invalid ({dens=}).")
        res["total_volume"] = tot_vol; res["density"] = dens
        sorted_y_s_arr = np.array(sorted(vp_prof.keys()))
        res["sorted_y_starts"] = sorted_y_s_arr; res["volume_profile"] = vp_prof
        _cut_start_t = time.time(); current_w_accum = 0.0; last_phys_cut_y = calc_ey_eff
        rel_indices = np.where((sorted_y_s_arr < calc_ey_eff) & (sorted_y_s_arr + slice_inc > calc_sy_eff))[0]
        if len(rel_indices)==0: raise ValueError("No profile slices in trim range for cutting.")
        y_s_rev_iter = sorted_y_s_arr[rel_indices][::-1]
        target_w_lower = target_w - w_tol; n_rev_sl = len(y_s_rev_iter)
        for rev_i, y_sl_s_abs in enumerate(y_s_rev_iter):
            prog_txt_area.text(f"Cutting (rev)... Slice at Y={y_sl_s_abs:.1f}mm ({rev_i+1}/{n_rev_sl})")
            y_sl_e_abs = y_sl_s_abs + slice_inc
            eval_part_s = max(y_sl_s_abs, calc_sy_eff); eval_part_e = min(y_sl_e_abs, last_phys_cut_y)
            if eval_part_e <= eval_part_s + FLOAT_EPSILON: continue
            vol_full_sl = vp_prof.get(y_sl_s_abs,0.0);
            if vol_full_sl <= 0: continue
            eval_part_len = eval_part_e - eval_part_s
            frac_in_eval = eval_part_len / slice_inc if slice_inc > FLOAT_EPSILON else 0.0
            wt_of_eval_part = (vol_full_sl * frac_in_eval) * dens
            if not (np.isfinite(wt_of_eval_part) and wt_of_eval_part >=0): wt_of_eval_part = 0.0
            if current_w_accum + wt_of_eval_part >= target_w_lower - FLOAT_EPSILON:
                cut_loc_y = eval_part_s; wt_carry_over = 0.0
                if not no_interp:
                    needed_w = target_w - current_w_accum
                    if wt_of_eval_part > FLOAT_EPSILON and needed_w > 0:
                        frac_to_take = max(0.0, min(1.0, needed_w / wt_of_eval_part))
                        cut_loc_y = eval_part_e - frac_to_take * eval_part_len
                        wt_carry_over = wt_of_eval_part * (1.0 - frac_to_take)
                    elif needed_w <= 0: cut_loc_y = eval_part_e; wt_carry_over = wt_of_eval_part
                cut_loc_y = max(cut_loc_y, calc_sy_eff)
                p_max_y_abs = last_phys_cut_y; p_min_y_abs = cut_loc_y
                p_len = max(0.0, p_max_y_abs - p_min_y_abs); act_p_wt = 0.0
                if p_len > FLOAT_EPSILON:
                    act_p_vol = recalculate_portion_volume(vp_prof, sorted_y_s_arr, slice_inc, p_min_y_abs, p_max_y_abs)
                    if np.isnan(act_p_vol): raise ValueError("Invalid vol in recalc.")
                    act_p_wt = act_p_vol * dens
                tmp_portions.append({"calc_max_y":p_max_y_abs, "calc_min_y":p_min_y_abs, "length":p_len, "weight":act_p_wt if np.isfinite(act_p_wt) else 0.0})
                last_phys_cut_y = p_min_y_abs - blade_thick
                current_w_accum = wt_carry_over if p_min_y_abs > calc_sy_eff + FLOAT_EPSILON else 0.0
            else: current_w_accum += wt_of_eval_part
            if last_phys_cut_y <= calc_sy_eff + FLOAT_EPSILON: break
        cut_time = time.time() - _cut_start_t
        if last_phys_cut_y > calc_sy_eff + FLOAT_EPSILON:
            w_max_y=last_phys_cut_y; w_min_y=calc_sy_eff; w_len=max(0.0,w_max_y-w_min_y); w_vol=0.0
            if w_len > FLOAT_EPSILON: w_vol=recalculate_portion_volume(vp_prof,sorted_y_s_arr,slice_inc,w_min_y,w_max_y)
            if np.isnan(w_vol): raise ValueError("Invalid vol for waste.")
            w_wt = w_vol*dens
            tmp_portions.append({"calc_max_y":w_max_y, "calc_min_y":w_min_y, "length":w_len, "weight":w_wt if np.isfinite(w_wt) else 0.0})
        final_ordered_ps = tmp_portions[::-1]
        for i,p_data in enumerate(final_ordered_ps):
            pmy, pmxy = p_data.get("calc_min_y"), p_data.get("calc_max_y")
            out_portions.append({"portion_num":i+1, "display_start_y":pmy if np.isfinite(pmy) else 0,
                               "display_end_y":pmxy if np.isfinite(pmxy) else 0, "length":p_data.get("length",0),
                               "weight":p_data.get("weight",0), "cut_y":pmxy if np.isfinite(pmxy) else 0})
        stat_msg = f"Calc complete. Found {len(out_portions)} portions."
        if out_portions: stat_msg += " (P1 is first physical piece/waste)."
    except OverflowError: stat_msg = "Err: Numerical overflow."
    except ValueError as ve: stat_msg = f"Err: {ve}"
    except Exception as ex: stat_msg = f"Unexpected err: {ex}"
    finally:
        tot_calc_t = time.time() - calc_t_start
        final_stat = f"{stat_msg} Total Time: {tot_calc_t:.2f}s (Profile: {vp_time:.2f}s, Cutting: {cut_time:.2f}s)"
        prog_txt_area.text(final_stat)
        if '_pb_area' in locals() and _pb_area is not None: _pb_area.empty()
        res["portions"]=out_portions; res["status"]=stat_msg; res["calc_time"]=tot_calc_t
    return res

def plot_3d_loaf(points_df, portions=None, title="Point Cloud", y_offset=0.0, camera_override=None): 
    if points_df is None or points_df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=points_df['x'], y=points_df['y'], z=points_df['z'], mode='markers',
                               marker=dict(size=1.5, color=points_df['z'], colorscale='YlOrBr', opacity=0.7,
                                           colorbar=dict(title='Height (Z)')), name='Point Cloud' ))
    if portions and len(portions) > 0:
        min_x_plot, max_x_plot = points_df['x'].min(), points_df['x'].max(); x_rng_plot = max(1.0, max_x_plot - min_x_plot)
        min_z_plot, max_z_plot = points_df['z'].min(), points_df['z'].max(); z_rng_plot = max(1.0, max_z_plot - min_z_plot)
        plane_x_coords = [min_x_plot - 0.05 * x_rng_plot, max_x_plot + 0.05 * x_rng_plot]
        plane_z_coords = [min_z_plot - 0.05 * z_rng_plot, max_z_plot + 0.05 * z_rng_plot]
        cut_y_locations_to_plot = []
        for i, p in enumerate(portions):
            if i < len(portions) - 1: 
                cut_y_val_for_plot = p.get('display_end_y', np.nan) + y_offset
                if np.isfinite(cut_y_val_for_plot): cut_y_locations_to_plot.append(cut_y_val_for_plot)
        for i, cut_y_loc in enumerate(cut_y_locations_to_plot):
             fig.add_trace(go.Mesh3d(
                x=[plane_x_coords[0],plane_x_coords[1],plane_x_coords[1],plane_x_coords[0]], y=[cut_y_loc]*4,
                z=[plane_z_coords[0],plane_z_coords[0],plane_z_coords[1],plane_z_coords[1]],
                opacity=0.4, color='red', alphahull=0, name=f"Cut {i+1} (Y={cut_y_loc:.1f})", showlegend=(i<3)))
    fig.update_layout(title=title, scene=dict(xaxis_title='Width (X,mm)',yaxis_title='Length (Y,mm)',zaxis_title='Height (Z,mm)',aspectmode='data'),margin=dict(l=0,r=0,b=0,t=40))
    return fig

def plot_area_profile(volume_profile, sorted_y_starts, slice_increment_mm, calc_start_y=None, calc_end_y=None): 
    if not volume_profile or len(sorted_y_starts) == 0: return go.Figure(layout=dict(title="Area Profile (No Data)", height=300))
    y_values_plot = sorted_y_starts
    area_values_plot = [(volume_profile.get(y_val, 0) / slice_increment_mm if slice_increment_mm > FLOAT_EPSILON else 0) for y_val in y_values_plot]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=y_values_plot, y=area_values_plot, mode='lines', name='Est. Area'))
    plot_min_y_boundary = y_values_plot[0] if len(y_values_plot) > 0 else 0
    plot_max_y_boundary = y_values_plot[-1] + slice_increment_mm if len(y_values_plot) > 0 else 0
    if calc_start_y is not None and calc_start_y > plot_min_y_boundary: fig.add_vrect(x0=plot_min_y_boundary, x1=calc_start_y, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="Start Trim")
    if calc_end_y is not None and calc_end_y < plot_max_y_boundary: fig.add_vrect(x0=calc_end_y, x1=plot_max_y_boundary, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="End Trim")
    fig.update_layout(title="Area Profile (Shaded=Trimmed Sections)", xaxis_title="Length (Y, mm - Relative to Calc Start)", yaxis_title="Est. Cross-Sectional Area (mm²)", margin=dict(l=20,r=20,t=40,b=20), height=300)
    return fig

def plot_cumulative_weight(volume_profile, sorted_y_starts, density_g_mm3, portions, target_weight, tolerance, calc_start_y=None, calc_end_y=None): 
    if not volume_profile or len(sorted_y_starts) == 0 or density_g_mm3 <= 0: return go.Figure(layout=dict(title="Cumulative Weight (No Data)", height=350))
    y_coords_plot = []; cumulative_weights_plot = []; current_weight_plot = 0.0
    if sorted_y_starts.size > 0:
        y_coords_plot.append(sorted_y_starts[0]); cumulative_weights_plot.append(0.0)
        for i, y_s_plot in enumerate(sorted_y_starts):
            slice_vol_plot = volume_profile.get(y_s_plot, 0.0); current_weight_plot += (slice_vol_plot * density_g_mm3)
            y_e_plot = sorted_y_starts[i+1] if (i + 1 < len(sorted_y_starts)) else (y_s_plot + (sorted_y_starts[1]-sorted_y_starts[0] if len(sorted_y_starts)>1 else 1.0)) 
            y_coords_plot.append(y_e_plot); cumulative_weights_plot.append(current_weight_plot)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=y_coords_plot, y=cumulative_weights_plot, mode='lines', name='Cumulative Weight'))
    fig.add_hline(y=target_weight, line_dash="dash", line_color="red", name="Target Weight")
    if tolerance > 0: fig.add_hline(y=target_weight + tolerance, line_dash="dot", line_color="orange", name="Target +Tolerance"); fig.add_hline(y=target_weight - tolerance, line_dash="dot", line_color="orange", name="Target -Tolerance")
    max_weight_for_plot = max(cumulative_weights_plot) if cumulative_weights_plot else target_weight * 1.1
    for i, p_plot in enumerate(portions):
        cut_y_plot_loc = p_plot.get('display_end_y', np.nan)
        if np.isfinite(cut_y_plot_loc) and i < len(portions) - 1:
             if p_plot.get('length', 0) > FLOAT_EPSILON : fig.add_vline(x=cut_y_plot_loc, line_dash="solid", line_color="grey", name=f"End of P{i+1}" if i<3 else None, showlegend=(i==0 and i < len(portions) -1 ))
    plot_min_y_boundary = y_coords_plot[0] if y_coords_plot else 0; plot_max_y_boundary = y_coords_plot[-1] if y_coords_plot else 0
    if calc_start_y is not None and calc_start_y > plot_min_y_boundary: fig.add_vrect(x0=plot_min_y_boundary, x1=calc_start_y, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="Start Trim")
    if calc_end_y is not None and calc_end_y < plot_max_y_boundary: fig.add_vrect(x0=calc_end_y, x1=plot_max_y_boundary, fillcolor="grey", opacity=0.15, layer="below", line_width=0, name="End Trim")
    fig.update_layout(title="Cumulative Weight Profile (Shaded=Trimmed)", xaxis_title="Length (Y, mm - Relative to Calc Start)", yaxis_title="Cumulative Weight (g)", yaxis_range=[0, max_weight_for_plot * 1.05], margin=dict(l=20,r=20,t=40,b=20), height=350)
    return fig

o3d_animation_globals = {
    "current_step": 0,
    "total_steps": 720,  
    "animation_type": None, 
    "pcd_center": np.array([0,0,0]),
    "initial_view_set": False
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

def start_o3d_visualization(points_df, animation_type_str): 
    if not _open3d_installed:
        st.error("Open3D library is not installed. This feature requires it.")
        return
    if points_df is None or points_df.empty:
        st.warning("No point cloud data to visualize.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_df[['x', 'y', 'z']].values)
    if not pcd.has_points():
        st.warning("Point cloud is empty after conversion to Open3D format.")
        return
    reset_o3d_animation_globals()
    o3d_animation_globals["animation_type"] = animation_type_str
    o3d_animation_globals["pcd_center"] = pcd.get_center()
    o3d_animation_globals["total_steps"] = 720 
    window_title = "Open3D Fly Around"
    st.info(f"Launching {window_title} in a new window. Close it to resume app interaction.")
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
        st.error(f"Failed to launch Open3D window: {e}")
    finally:
        st.info("Open3D window closed.")
        reset_o3d_animation_globals()

def create_o3d_plane_mesh(y_pos, x_min, x_max, z_min, z_max, color, plane_thickness_along_y=1.0):
    if not _open3d_installed: return None
    plane_extent_x = x_max - x_min
    plane_extent_z = z_max - z_min
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=plane_extent_x, 
                                                    height=plane_thickness_along_y, 
                                                    depth=plane_extent_z) 
    translate_x = x_min 
    translate_y = y_pos - (plane_thickness_along_y / 2) 
    translate_z = z_min 
    mesh_box.translate([translate_x, translate_y, translate_z])
    mesh_box.paint_uniform_color(color)
    return mesh_box

def launch_o3d_viewer_with_cuts(points_df, portions_data, 
                                calc_start_y_from_res, calc_end_y_from_res, 
                                y_offset_to_add_back):
    if not _open3d_installed:
        st.error("Open3D library is not installed. This feature requires it.")
        return
    if points_df is None or points_df.empty:
        st.warning("No point cloud data to visualize.")
        return
    if not portions_data:
        st.warning("No portion data available to visualize cuts.")
        return

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points_df[['x', 'y', 'z']].values)
    if not pcd_o3d.has_points():
        st.warning("Point cloud is empty for Open3D visualization.")
        return

    geometries = [pcd_o3d]
    x_min_viz, x_max_viz = points_df['x'].min(), points_df['x'].max()
    z_min_viz, z_max_viz = points_df['z'].min(), points_df['z'].max()
    x_padding = (x_max_viz - x_min_viz) * 0.05; z_padding = (z_max_viz - z_min_viz) * 0.05
    plane_x_min = x_min_viz - x_padding; plane_x_max = x_max_viz + x_padding
    plane_z_min = z_min_viz - z_padding; plane_z_max = z_max_viz + z_padding
    color_portion_cut = [0.8, 0.2, 0.2]; color_trim_cut = [0.2, 0.2, 0.8]   
    plane_thickness_val = max(1.0, (points_df['y'].max() - points_df['y'].min()) * 0.002) 

    if calc_start_y_from_res is not None:
        y_plane_pos_start_trim = calc_start_y_from_res + y_offset_to_add_back
        start_trim_plane = create_o3d_plane_mesh(y_plane_pos_start_trim, plane_x_min, plane_x_max, plane_z_min, plane_z_max, color_trim_cut, plane_thickness_val)
        if start_trim_plane: geometries.append(start_trim_plane)
    if calc_end_y_from_res is not None:
        y_plane_pos_end_trim = calc_end_y_from_res + y_offset_to_add_back
        end_trim_plane = create_o3d_plane_mesh(y_plane_pos_end_trim, plane_x_min, plane_x_max, plane_z_min, plane_z_max, color_trim_cut, plane_thickness_val)
        if end_trim_plane: geometries.append(end_trim_plane)
        
    for i, p_item in enumerate(portions_data):
        cut_y_val_calc = p_item.get('cut_y') 
        if cut_y_val_calc is not None:
            is_start_trim_cut = (calc_start_y_from_res is not None and abs(cut_y_val_calc - calc_start_y_from_res) < FLOAT_EPSILON)
            is_end_trim_cut = (calc_end_y_from_res is not None and abs(cut_y_val_calc - calc_end_y_from_res) < FLOAT_EPSILON)
            if not is_start_trim_cut and not is_end_trim_cut:
                 if i < len(portions_data) -1 : 
                    effective_cut_y_orig_space = cut_y_val_calc + y_offset_to_add_back
                    portion_cut_plane = create_o3d_plane_mesh(effective_cut_y_orig_space, plane_x_min, plane_x_max, plane_z_min, plane_z_max, color_portion_cut, plane_thickness_val)
                    if portion_cut_plane: geometries.append(portion_cut_plane)

    st.info("Launching Open3D viewer with cuts. Close the Open3D window to resume app interaction.")
    try:
        o3d.visualization.draw_geometries(
            geometries, window_name="Open3D - Point Cloud with Cuts",
            width=1280, height=720, point_show_normal=False
        )
    except Exception as e: st.error(f"Failed to launch Open3D window: {e}")
    finally: st.info("Open3D window closed.")

st.set_page_config(layout="wide", page_title="Point Cloud Portion Calc v1.4.12", page_icon="🧀")

default_persistent_states = { 
    'point_cloud_data': None, 'data_origin': None, 'last_file_id': None, 'calc_results': None,
    'ror_applied_cloud_id': None, 'ror_estimated_radius_msg': None, 'ror_filter_applied_msg': None,
    'previous_data_source_tracker': None, 'data_source': "Generate Test Data",
    'total_weight': DEFAULT_TOTAL_WEIGHT, 'target_weight': DEFAULT_TARGET_WEIGHT,
    'weight_tolerance': DEFAULT_WEIGHT_TOLERANCE, 'no_interp': False,
    'slice_thickness': DEFAULT_SLICE_THICKNESS, 'start_trim': DEFAULT_START_TRIM,
    'end_trim': DEFAULT_END_TRIM, 'blade_thickness': DEFAULT_BLADE_THICKNESS,
    'top_down_scan': False, 'flat_bottom': False, 'voxel_size': DEFAULT_VOXEL_SIZE,
    'resolution': DEFAULT_RESOLUTION, 'ror_nb_points': 5, 'ror_radius_val': 0.1,
    'ror_k_radius_est': 20, 'ror_radius_mult_est': 2.0, 'enable_auto_downsample': True,
    'auto_downsample_threshold': DEFAULT_AUTO_DOWNSAMPLE_THRESHOLD,
    'enable_y_normalization': True,
    'y_min_of_displayed_cloud': 0.0,
    'density_source': "Calculate from Total Weight & Volume",
    'direct_density_g_cm3': DEFAULT_DIRECT_DENSITY_G_CM3,
}
for key_init, value_init in default_persistent_states.items():
    if key_init not in st.session_state: st.session_state[key_init] = value_init

with st.sidebar: 
    st.header("🧀 Parameters")
    st.subheader("1. Data Source")
    data_source_options = ("Generate Test Data", "Upload File")
    try: default_ds_index = data_source_options.index(st.session_state.data_source)
    except ValueError: default_ds_index = 0
    st.radio("Select Source:", options=data_source_options, index=default_ds_index, label_visibility="collapsed", key="data_source")
    uploaded_file_sb = None
    file_types_sb_main = ['csv', 'xyz'] + (['pcd', 'ply'] if _open3d_installed else []) + ['xlsx', 'xls']
    if st.session_state.data_source == "Upload File":
        uploaded_file_sb = st.file_uploader("Choose file", type=file_types_sb_main, key="uploader_widget")
        if not _open3d_installed and any(ft in ['pcd','ply'] for ft in file_types_sb_main): st.caption("Install `open3d` for .pcd/.ply.")
        st.caption("Excel: 'x','y','z' cols on 1st sheet.")
    else: st.caption("_(Using generated test data)_")

    
    if st.session_state.point_cloud_data is not None and not st.session_state.point_cloud_data.empty and _open3d_installed and st.session_state.data_source == "Upload File" and uploaded_file_sb is not None:
        st.subheader("Convert & Download Current Cloud")
        
        if st.button("Download as Binary PLY"):
            with st.spinner("Preparing Binary PLY download..."):
                try:
                    o3d_pcd_to_save = o3d.geometry.PointCloud()
                    o3d_pcd_to_save.points = o3d.utility.Vector3dVector(st.session_state.point_cloud_data[['x','y','z']].values)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp_ply:
                        o3d.io.write_point_cloud(tmp_ply.name, o3d_pcd_to_save, write_ascii=False)
                        tmp_ply_path = tmp_ply.name
                    
                    with open(tmp_ply_path, "rb") as fp:
                        st.download_button(
                            label="Click to Download Binary PLY",
                            data=fp,
                            file_name="converted_cloud.ply",
                            mime="application/octet-stream" 
                        )
                    if os.path.exists(tmp_ply_path): os.remove(tmp_ply_path)
                except Exception as e_ply_save:
                    st.error(f"Error creating PLY: {e_ply_save}")

    st.subheader("2. Weight & Density Settings")

    def update_density_defaults():
        if st.session_state.density_source == "Calculate from Total Weight & Volume":
            st.session_state.total_weight = DEFAULT_TOTAL_WEIGHT
        else:
            st.session_state.direct_density_g_cm3 = DEFAULT_DIRECT_DENSITY_G_CM3
            st.session_state.total_weight = DEFAULT_TOTAL_WEIGHT

    st.radio(
        "Density Source:",
        options=["Calculate from Total Weight & Volume", "Input Directly"],
        key="density_source",
        horizontal=True,
        on_change=update_density_defaults
    )

    if st.session_state.density_source == "Calculate from Total Weight & Volume":
        st.number_input("Total Loaf Wt (g)", min_value=0.01, key="total_weight", step=50.0, format="%.2f")
        st.caption("Density will be calculated.")
    else:
        st.number_input("Direct Density (g/cm³)", min_value=0.01, key="direct_density_g_cm3", step=0.01, format="%.3f", help="Enter density.")
        st.caption("Density is input directly. Water is 1 g/cm³, most cheeses are 1.0 to 1.2 g/cm³")
        st.number_input("Total Loaf Wt (g) (Informational)", min_value=0.0, key="total_weight", step=50.0, format="%.2f", help="Reference only.")

    st.number_input("Target Portion Wt (g)", min_value=0.01, key="target_weight", step=10.0, format="%.2f")
    st.number_input("Target Wt Tolerance (+/- g)", min_value=0.0, key="weight_tolerance", step=0.5, format="%.1f")
    st.subheader("3. Calculation Settings")
    st.toggle("Cut Portions Always Equal Or Greater Than Target Weight (No Interpolation)", key="no_interp")
    st.slider("Calc. Slice Thickness (mm)", 0.1, 10.0, key="slice_thickness", step=0.1, format="%.1f")
    st.number_input("Start Trim (mm)", 0.0, key="start_trim", step=1.0, format="%.1f")
    st.number_input("End Trim (mm)", 0.0, key="end_trim", step=1.0, format="%.1f")
    st.number_input("Blade Thickness / Kerf (mm)", 0.0, key="blade_thickness", step=0.1, format="%.1f")
    st.subheader("4. Advanced & Simulation")
    st.caption("Presums a 4 profiler setup unless 'Top-Down Only' or 'Top & Side Only' is checked.")
    st.toggle("Top-Down Only (1 Profiler)", key="top_down_scan")
    st.toggle("Top & Side Only (2 Profilers)", key="flat_bottom", disabled=st.session_state.top_down_scan)
    st.number_input("Voxel Downsample Size (mm)", 0.0, key="voxel_size", step=0.25, format="%.2f")
    if st.session_state.voxel_size > 0 and not _open3d_installed: st.warning("Voxel downsampling needs `open3d`.")
    st.checkbox("Enable Auto-Downsample", key="enable_auto_downsample", help="If point count exceeds threshold, automatically downsample using Random Sampling.")
    st.number_input(
        "Auto-Downsample Threshold (points)",
        min_value=50000, max_value=5000000, step=50000,
        key="auto_downsample_threshold",
        help="If 'Enable Auto-Downsample' is checked, clouds with more points than this threshold will be downsampled.",
        disabled=not st.session_state.enable_auto_downsample
    )
    st.checkbox("Normalize Y-coords before Calculation", key="enable_y_normalization", help="Shift Y to start near Y=0 for calc.")
    st.number_input("Generated Test Data Cloud Resolution (mm)", 0.01, key="resolution", step=0.1, format="%.2f")
    st.subheader("5. Point Cloud Filtering (Optional)")
    st.caption("Applied after loading & downsampling. Requires 'open3d'.")
    with st.expander("Radius Outlier Removal (ROR)", expanded=False):
        if not _open3d_installed: st.warning("ROR requires `open3d`.")
        else:
            st.slider("ROR: Min pts in radius", 1, 100, key="ror_nb_points")
            ror_c1, ror_c2 = st.columns(2)
            with ror_c1: st.slider("k for est.", 1, 50, key="ror_k_radius_est")
            with ror_c2: st.slider("Radius mult.", 0.1, 10.0, key="ror_radius_mult_est", step=0.1)
            if st.button("Estimate ROR Radius", use_container_width=True, key="ror_est_btn_widget"):
                if st.session_state.point_cloud_data is not None and not st.session_state.point_cloud_data.empty:
                    pcd_est_ror = o3d.geometry.PointCloud(); pcd_est_ror.points = o3d.utility.Vector3dVector(st.session_state.point_cloud_data[['x','y','z']].values)
                    if not pcd_est_ror.has_points(): st.session_state.ror_estimated_radius_msg = "Cloud empty."
                    else:
                        est_r, msg_r = estimate_ror_radius_util_o3d(pcd_est_ror, st.session_state.ror_k_radius_est, st.session_state.ror_radius_mult_est)
                        st.session_state.ror_estimated_radius_msg = msg_r;
                        if est_r is not None: st.session_state.ror_radius_val = float(f"{est_r:.4f}")
                else: st.session_state.ror_estimated_radius_msg = "Load cloud first."
            if st.session_state.ror_estimated_radius_msg: st.caption(st.session_state.ror_estimated_radius_msg)
            st.number_input("ROR: Search Radius (mm)", 0.0001, key="ror_radius_val", step=0.001, format="%.4f")
            if st.button("Apply ROR Filter", type="secondary", use_container_width=True, key="ror_apply_btn_widget"):
                if st.session_state.point_cloud_data is not None and not st.session_state.point_cloud_data.empty:
                    if st.session_state.ror_radius_val > 0:
                        try:
                            pcd_apply_ror = o3d.geometry.PointCloud(); pcd_apply_ror.points = o3d.utility.Vector3dVector(st.session_state.point_cloud_data[['x','y','z']].values)
                            if not pcd_apply_ror.has_points(): st.warning("ROR: Cloud empty.")
                            else:
                                n_b = len(pcd_apply_ror.points)
                                _, ind_apply = pcd_apply_ror.remove_radius_outlier(st.session_state.ror_nb_points, st.session_state.ror_radius_val)
                                pcd_filt = pcd_apply_ror.select_by_index(ind_apply)
                                n_a = len(pcd_filt.points); n_rem = n_b - n_a
                                st.session_state.point_cloud_data = pd.DataFrame(np.asarray(pcd_filt.points), columns=['x','y','z'])
                                st.session_state.ror_filter_applied_msg = (f"ROR: Removed {n_rem} pts. Now: {n_a:,}.")
                                st.session_state.ror_applied_cloud_id = st.session_state.last_file_id if st.session_state.last_file_id else "generated_data"
                                st.toast(st.session_state.ror_filter_applied_msg, icon="🧹")
                                st.session_state.calc_results = None; st.rerun()
                        except Exception as e_ror_app: st.error(f"ROR Error: {e_ror_app}"); st.session_state.ror_filter_applied_msg = f"ROR Error: {e_ror_app}"
                    else: st.warning("ROR radius > 0.")
                else: st.warning("Load cloud first for ROR.")
        curr_cloud_id = st.session_state.last_file_id if st.session_state.last_file_id else "generated_data"
        if st.session_state.ror_filter_applied_msg and st.session_state.ror_applied_cloud_id == curr_cloud_id:
             st.success(st.session_state.ror_filter_applied_msg)
    st.divider()
    calc_button_main_ui = st.button("Calculate Portions", type="primary", use_container_width=True)
    st.caption("P1 is first physical piece (waste).\nClick 'Calculate Portions'.")
    
    st.markdown("---") 
    if st.button("🔄 Reset All to Defaults", key="reset_all_btn", use_container_width=True, type="secondary", help="Resets all options and reloads/regenerates the initial point cloud."):
        for key_to_clear in list(st.session_state.keys()): 
            if key_to_clear in default_persistent_states:
                 del st.session_state[key_to_clear]
            elif key_to_clear in ['calc_results', 'ror_filter_applied_msg', 'ror_applied_cloud_id', 'ror_estimated_radius_msg', 'previous_data_source_tracker', 'data_origin', 'last_file_id', 'point_cloud_data', 'y_min_of_displayed_cloud']:
                 del st.session_state[key_to_clear]
        if 'o3d_animation_globals' in globals(): 
            reset_o3d_animation_globals()
        st.toast("All settings cleared. Defaults will apply on refresh. Point cloud will reload/regenerate.", icon="🧼")
        st.rerun()

st.title(f"🔪 Point Cloud Cutting Portion Calculator")
st.markdown("<p style='font-size:18px;'>Created By Shaun Harris</p>", unsafe_allow_html=True)
st.markdown("Calculates cuts. **Portion 1** is the first physical piece (waste).")

with st.expander("ℹ️ Help / App Information", expanded=False):
    st.markdown("""
    This application helps you calculate optimal cutting portions based on 3D point cloud data.
    ---
    #### **How to Use:**
    **1. Sidebar Parameters:** The sidebar on the left contains all the input parameters.
    *   **1. Data Source:**
        *   `Generate Test Data`: Creates a synthetic loaf. `Scanner Resolution` (Advanced) affects it.
        *   `Upload File`: Upload CSV, XYZ, PCD, PLY, XLSX, XLS. For CSV/Excel: use 'x', 'y', 'z' columns (case-insensitive) on the first sheet, or first 3 cols will be assumed. PCD/PLY need `open3d`.
    *   **2. Weight & Density Settings:**
        *   `Density Source`: Choose `Calculate from Total Weight & Volume` (app estimates density) or `Input Directly` (provide known density in g/cm³).
        *   `Total Loaf Wt (g)`: Used if density is calculated. Informational if density is input directly.
        *   `Target Portion Wt (g)`: Desired weight for main portions.
        *   `Target Wt Tolerance (+/- g)`: Allowable deviation for highlighting results. Calculation aims for `Target Wt - Tolerance` as a minimum before interpolation.
    *   **3. Calculation Settings:**
        *   `Cut > Target (No Interpolation)`: If ON, portions are >= `Target Wt - Tolerance`. If OFF (Interpolation ON), aims for exact `Target Wt`.
        *   `Calc. Slice Thickness (mm)`: Thickness of virtual slices for volume analysis.
        *   `Start/End Trim (mm)`: Amount trimmed from front/rear (considered waste).
        *   `Blade Thickness / Kerf (mm)`: Material lost per cut.
    *   **4. Advanced & Simulation:**
        *   `Top-Down Scan Only`: Simulates a scan capturing only top surface and width.
        *   `Top & Side Scan Only`: Assumes a flat bottom (Z=0 for calc). Uses XZ convex hull.
        *   `Voxel Downsample Size (mm)`: If > 0, downsamples cloud to this voxel grid size (needs `open3d`).
        *   `Enable Auto-Downsample`: If point count exceeds threshold, automatically downsample using Random Sampling.
        *   `Auto-Downsample Threshold (points)`: User-defined threshold for when auto-downsampling kicks in (if enabled).
        *   `Normalize Y-coords before Calculation`: Shifts loaf Y-coords to start near Y=0 for internal calculation.
        *   `Scanner Resolution (mm)`: Primarily for test data generation noise/density.
    *   **5. Point Cloud Filtering (ROR):** (Radius Outlier Removal - needs `open3d`)
        *   Removes points with few neighbors in a radius. `Estimate ROR Radius` helps find a good `Search Radius`.
    *   **Calculate Portions Button:** Starts the calculation.
    *   **Reset All to Defaults Button:** Resets all sidebar options and reloads/regenerates the point cloud.
    ---
    #### **Interpreting the Display:**
    *   **Point Cloud Input Metrics:** Shows current point count and estimated dimensions.
    *   **Interactive 3D Inspection (External Open3D Window):**
        *   `Fly Around Loaf (Open3D)`: Opens a new window to orbit the point cloud (requires `open3d`). This feature works best when running the script locally.
    *   **Current Point Cloud (Static Preview):** A static 3D plot. Calculated cuts are overlaid (red planes).
    ---
    #### **Portioning Results:**
    *   **Status Message:** Calculation success, errors, or warnings.
    *   **Results Summary Tab:**
        *   `Calculated Portions Table`: Portion #1 is the first physical piece from the front. Highlights: Yellow (P1), Red (outside tolerance).
        *   `Download Results CSV`: Saves table data.
        *   `View Cuts in 3D (Open3D)`: Shows cloud with trim planes (blue) and portion cuts (red) in a new window (requires `open3d`). This feature works best when running the script locally.
    *   `Metrics`: Target/Tolerance, P1 weight, Total Calculated Weight, Density/Volume used.
    *   **Analysis Plots Tab:**
        *   `Area Profile`: Estimated cross-sectional area along loaf length. Shaded regions = trims.
        *   `Cumulative Weight Profile`: Weight accumulation along loaf. Target lines and cuts overlaid.
    ---
    #### **Calculation Process Overview:**
    1.  **Preprocessing:** Load, Y-Normalize (opt), Voxel Downsample (opt), Auto-Downsample (opt), ROR (opt).
    2.  **Volume Profiling:** Loaf is virtually sliced along Y. Slice Volume = XZ Area * Slice Thickness.
    3.  **Density:** Calculated from Total Wt/Est. Volume, or input directly.
    4.  **Cutting (Reversed):** Starts from loaf's rear (`calc_ey_eff`). Accumulates weight moving to front. Cuts when `Target Wt - Tolerance` is met/exceeded. Interpolation (if ON) refines cut for `Target Wt`. Blade thickness shifts next portion's start.
    5.  **Portion Refinement:** Final portion weights are recalculated using precise cut boundaries and the volume profile.
    6.  **Output Ordering:** Portions are listed from front to rear.
    ---        
    """)

    st.subheader("🎬 Video: Convex Hull Scanning Algorithm Explained")
    st.markdown("""
    #### **Convex Hull for Area Estimation:** 
    *   For each thin slice of the point cloud (unless "Top-Down Scan" is selected), the application projects the points within that slice onto the XZ plane. It then calculates the **Convex Hull** of these 2D points.
    *   What is a Convex Hull?* Imagine stretching a rubber band around the outermost points in that 2D slice; the shape enclosed by the rubber band is the convex hull. It's the smallest convex polygon that contains all the points.
    *   The area of this convex hull is used as an approximation of the cheese's cross-sectional area for that specific slice. This method is effective for capturing the general shape of the loaf's cross-section, especially if the loaf is reasonably convex.
    *   If the "Top & Side Scan Only (Flat Bottom)" option is active, points at Z=0 are added to the slice's XZ points before computing the hull to ensure the bottom is flat.  
    ---   
    """)
    col1, col_video, col3 = st.columns([0.5, 0.5, 0.5]) 
    with col_video:
        try:
            video_file_path = "C:\\ImagePortion\\Convex_Hull _Graham_Scan_Algorithm.mp4" 
            if os.path.exists(video_file_path):
                video_file = open(video_file_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                video_file.close()
            else:
                st.warning(f"Video file not found at: {video_file_path}")
        except Exception as e:
            st.error(f"Error loading video: {e}")

refresh_data_flag_main_ui = False 
previous_data_source_tracker = st.session_state.get("previous_data_source_tracker", None)
if st.session_state.data_source != previous_data_source_tracker:
    refresh_data_flag_main_ui = True; st.session_state.previous_data_source_tracker = st.session_state.data_source
if st.session_state.data_source == "Generate Test Data":
    if st.session_state.get("data_origin") != "Generated" or st.session_state.get("point_cloud_data") is None: refresh_data_flag_main_ui = True
elif st.session_state.data_source == "Upload File" and uploaded_file_sb is not None:
    current_file_id_main_ui = f"{uploaded_file_sb.name}-{uploaded_file_sb.size}"
    if current_file_id_main_ui != st.session_state.get("last_file_id"):
        refresh_data_flag_main_ui = True
elif st.session_state.data_source == "Upload File" and st.session_state.get("last_file_id") is None and st.session_state.get("point_cloud_data") is None:
    pass

if refresh_data_flag_main_ui: # BLOCK 1
    st.toast("Refreshing data source...", icon="🔄")
    st.session_state.calc_results = None; st.session_state.ror_filter_applied_msg = None
    st.session_state.ror_applied_cloud_id = None; st.session_state.ror_estimated_radius_msg = None
    temp_df_load_main_ui = None; current_step = 0; total_steps = 1
    if st.session_state.data_source == "Generate Test Data":
         with st.spinner("Generating test data..."):
            temp_df_load_main_ui = generate_test_point_cloud(resolution=st.session_state.resolution, seed=42)
            st.session_state.original_point_count = len(temp_df_load_main_ui) if temp_df_load_main_ui is not None else 0
            st.session_state.data_origin = "Generated"; st.session_state.last_file_id = "generated_data"
            if temp_df_load_main_ui is not None: st.toast(f"Generated {len(temp_df_load_main_ui):,} pts.", icon="✨")
            else: st.error("Failed to generate test data.")
    elif st.session_state.data_source == "Upload File" and uploaded_file_sb is not None:
        if uploaded_file_sb.size > 5*1024*1024 and st.session_state.enable_auto_downsample : total_steps +=1 
        if st.session_state.voxel_size > 0: total_steps += 1
        current_step += 1; spinner_msg_load = f"Step {current_step}/{total_steps}: Reading {uploaded_file_sb.name}..."
        with st.spinner(spinner_msg_load): temp_df_load_main_ui = load_point_cloud(uploaded_file_sb)
        if temp_df_load_main_ui is not None:
            st.session_state.original_point_count = len(temp_df_load_main_ui)
            st.session_state.data_origin = f"Uploaded: {uploaded_file_sb.name}"; st.session_state.last_file_id = f"{uploaded_file_sb.name}-{uploaded_file_sb.size}"
            if st.session_state.enable_auto_downsample and len(temp_df_load_main_ui) > st.session_state.auto_downsample_threshold: 
                current_step +=1; spinner_msg_auto_ds = f"Step {current_step}/{total_steps}: Auto-downsampling (Random) as point count {len(temp_df_load_main_ui):,} > {st.session_state.auto_downsample_threshold:,}..."
                with st.spinner(spinner_msg_auto_ds):
                    if _open3d_installed:
                        try:
                            pcd_auto_ds_ui=o3d.geometry.PointCloud(); pcd_auto_ds_ui.points=o3d.utility.Vector3dVector(temp_df_load_main_ui[['x','y','z']].values) 
                            ratio_auto_ui = st.session_state.auto_downsample_threshold / len(pcd_auto_ds_ui.points); 
                            ratio_auto_ui = max(min(ratio_auto_ui, 1.0), 0.00001) 
                            pcd_down_auto_ui = pcd_auto_ds_ui.random_down_sample(ratio_auto_ui); 
                            method_auto_ui = f"Random (target ratio {ratio_auto_ui:.3f})"
                            if pcd_down_auto_ui and len(pcd_down_auto_ui.points)>0: 
                                temp_df_load_main_ui=pd.DataFrame(np.asarray(pcd_down_auto_ui.points), columns=['x','y','z'])
                                st.toast(f"Auto-downsampled to {len(temp_df_load_main_ui):,} pts ({method_auto_ui}).",icon="📉")
                            else: st.warning("Auto-downsampling (Random) resulted in empty cloud. Using original.")
                        except Exception as e_auto_ui: st.warning(f"Auto-downsample (Random) error: {e_auto_ui}. Using original.")
                    else: st.warning(f"Large cloud but `open3d` not for auto-downsampling (step skipped).")
        else: st.session_state.data_origin = "Upload Failed"; st.session_state.last_file_id = None
    
    processed_df_session_ui = temp_df_load_main_ui
    if temp_df_load_main_ui is not None and not temp_df_load_main_ui.empty and st.session_state.voxel_size > 0:
        if _open3d_installed:
            current_step +=1; spinner_msg_voxel = f"Step {current_step}/{total_steps}: Voxel Downsampling..." 
            with st.spinner(spinner_msg_voxel):
                pcd_vox_ui=o3d.geometry.PointCloud(); pcd_vox_ui.points=o3d.utility.Vector3dVector(temp_df_load_main_ui[['x','y','z']].values)
                if not pcd_vox_ui.has_points(): st.warning("Cloud empty before voxel downsample.")
                else:
                    down_pcd_vox_ui=pcd_vox_ui.voxel_down_sample(st.session_state.voxel_size); down_pts_vox_ui=np.asarray(down_pcd_vox_ui.points)
                    if down_pts_vox_ui.shape[0]>0: processed_df_session_ui=pd.DataFrame(down_pts_vox_ui,columns=['x','y','z']); st.toast(f"Voxel downsampled to {len(processed_df_session_ui):,} pts.",icon="📦")
                    else: st.warning("Voxel downsample gave 0 pts. Using pre-voxel data.")
        else: st.warning("Voxel downsample selected but `open3d` not found.")
    st.session_state.point_cloud_data = processed_df_session_ui
    st.rerun()

# BLOCK 2: Display Point Cloud Info and 3D Plot 
st.subheader("Point Cloud Input")
points_df_disp_ui = st.session_state.get('point_cloud_data')
if points_df_disp_ui is not None and not points_df_disp_ui.empty:
    st.markdown(f"**Source:** {st.session_state.get('data_origin', 'N/A')}")
    col1_met_ui, col2_met_ui, col3_met_ui = st.columns(3)
    try:
        if np.all(np.isfinite(points_df_disp_ui[['x','y','z']].values)):
            min_dims_ui,max_dims_ui,dims_ui=points_df_disp_ui.min(),points_df_disp_ui.max(),points_df_disp_ui.max()-points_df_disp_ui.min()
            col1_met_ui.metric("Current Points",f"{len(points_df_disp_ui):,} pts",)
            if st.session_state.enable_auto_downsample and st.session_state.original_point_count > st.session_state.auto_downsample_threshold:
                st.markdown(f"Before Downsample: {st.session_state.get('original_point_count', 0):,} pts")
            col2_met_ui.metric("Loaf Length (Y)", f"{dims_ui.get('y', np.nan):.2f} mm")
            col3_met_ui.metric("Est. Max Width (X)", f"{dims_ui.get('x', np.nan):.2f} mm")
        else: col1_met_ui.metric("Current Points",f"{len(points_df_disp_ui):,}"); col2_met_ui.warning("Non-finite data in cloud.")
    except Exception as e_met_ui: st.warning(f"Dimension display error: {e_met_ui}")

    st.markdown("---")
    st.subheader("🎬 Interactive 3D Inspection (External Open3D Window. Needs To Be Ran Locally)")
    st.caption("Launches a new window with Open3D for animated inspection. Close the Open3D window to return to the app.")
    
    if st.button("🚁 Fly Around Loaf (Open3D)", key="o3d_fly_around_btn", use_container_width=True, disabled=not _open3d_installed):
        start_o3d_visualization(points_df_disp_ui, 'o3d_fly_around')

    if not _open3d_installed: 
        st.warning("Open3D library is not installed or could not be imported. External inspection features are disabled.", icon="⚠️")

    with st.spinner("Generating 3D plot for static preview..."):
        plot_title = "Cloud with Cuts" if st.session_state.get('calc_results') and st.session_state.calc_results.get("portions") else "Current Point Cloud (Static Preview)"
        plot_portions = (st.session_state.get('calc_results') or {}).get("portions")
        y_offset_for_plot_display = (st.session_state.get('calc_results') or {}).get("y_offset_for_plot", 0.0)
        fig_3d_main_plot = plot_3d_loaf(points_df_disp_ui, portions=plot_portions, title=plot_title,
                                        y_offset=y_offset_for_plot_display, camera_override=None)
        st.plotly_chart(fig_3d_main_plot, use_container_width=True, key="main_3d_plot_static")

elif st.session_state.data_source == "Upload File" and uploaded_file_sb is None and not st.session_state.get("last_file_id"):
    st.info("Upload a point cloud file using the sidebar.")
elif st.session_state.data_source == "Generate Test Data" and (points_df_disp_ui is None or points_df_disp_ui.empty) and not calc_button_main_ui:
     st.info("Click 'Calculate Portions' or change parameters to generate test data.")
else:
    if not st.session_state.get('point_cloud_data'):
       st.info("Generate or upload point cloud data via sidebar to begin.")

# BLOCK 3: Calculation Trigger 
if calc_button_main_ui:
    points_df_for_calc_main = st.session_state.get('point_cloud_data')
    if points_df_for_calc_main is not None and not points_df_for_calc_main.empty:
        if np.all(np.isfinite(points_df_for_calc_main[['x','y','z']].values)):
            y_min_for_normalization = 0.0
            df_to_calculate_with = points_df_for_calc_main
            if st.session_state.enable_y_normalization:
                df_to_calculate_with = points_df_for_calc_main.copy()
                y_min_for_normalization = df_to_calculate_with['y'].min()
                df_to_calculate_with['y'] = df_to_calculate_with['y'] - y_min_for_normalization
                st.toast(f"Calc using Y-coords normalized by {-y_min_for_normalization:.2f}mm.", icon="📏")
            else:
                st.toast("Calculation using original Y-coordinates.", icon="📐")
            with st.spinner("Calculating cuts... This may take a moment..."):
                eff_fb_calc_ui = st.session_state.flat_bottom or st.session_state.top_down_scan
                input_direct_density_g_mm3 = None
                if st.session_state.density_source == "Input Directly":
                    direct_density_g_cm3_val = st.session_state.get('direct_density_g_cm3', 0.0)
                    if direct_density_g_cm3_val > FLOAT_EPSILON:
                        input_direct_density_g_mm3 = direct_density_g_cm3_val / 1000.0 
                        st.toast(f"Using direct density: {direct_density_g_cm3_val:.3f} g/cm³.", icon="🔬")

                    else:
                        st.error("Direct density selected, but value is invalid."); st.session_state.calc_results = {"status": "Error: Invalid direct density."}; st.rerun()
                else: 
                    if st.session_state.total_weight <= FLOAT_EPSILON:
                        st.error("Density from Total Weight selected, but Total Loaf Wt is invalid."); st.session_state.calc_results = {"status": "Error: Invalid Total Wt for density calc."}; st.rerun()
                    st.toast("Density from total weight and volume.", icon="⚖️")
                st.session_state.calc_results = calculate_cut_portions_reversed(
                    df_to_calculate_with, st.session_state.total_weight, st.session_state.target_weight,
                    st.session_state.slice_thickness, st.session_state.no_interp,
                    eff_fb_calc_ui, st.session_state.top_down_scan,
                    st.session_state.blade_thickness, st.session_state.weight_tolerance,
                    st.session_state.start_trim, st.session_state.end_trim,
                    direct_density_g_mm3=input_direct_density_g_mm3 
                )
                if st.session_state.calc_results: st.session_state.calc_results["y_offset_for_plot"] = y_min_for_normalization
            st.rerun()
        else:
            st.error("Cannot calc - cloud has non-finite values."); st.session_state.calc_results = {"status": "Error: Non-finite values in cloud."}
    else: st.warning("No point cloud data for calculation.")

# BLOCK 4: Display Results 
st.subheader("Portioning Results")
calc_res_disp_ui = st.session_state.get('calc_results')
if calc_res_disp_ui:
    stat_msg_ui = calc_res_disp_ui.get("status", "Status unknown")
    portions_disp_ui = calc_res_disp_ui.get("portions", [])
    density_source_msg_ui = calc_res_disp_ui.get("density_source_message", "")
    if "Error:" in stat_msg_ui: st.error(stat_msg_ui)
    elif "Warning:" in stat_msg_ui: st.warning(stat_msg_ui)
    else: st.info(stat_msg_ui)
    if density_source_msg_ui and "Error:" not in stat_msg_ui : st.caption(density_source_msg_ui)
    if portions_disp_ui:
        def highlight_style_func_ui(df_style_ui):
            target_style_ui=st.session_state.target_weight; tol_style_ui=st.session_state.weight_tolerance
            styles_ui=pd.DataFrame('', index=df_style_ui.index, columns=df_style_ui.columns)
            if not df_style_ui.empty: styles_ui.iloc[0]='background-color: #FFF3CD; font-weight: bold;'
            if tol_style_ui > 0 and len(df_style_ui)>1:
                 try: 
                     wt_col_idx_ui=df_style_ui.columns.get_loc("Est. Weight (g)") 
                     wts_num_ui=pd.to_numeric(df_style_ui.iloc[1:,wt_col_idx_ui],errors='coerce') 
                     outside_mask_ui=((wts_num_ui < target_style_ui-tol_style_ui)|(wts_num_ui > target_style_ui+tol_style_ui)) 
                     styles_ui.iloc[1:][outside_mask_ui.fillna(False).values]='background-color: #F8D7DA;'
                 except (KeyError, Exception): pass 
            return styles_ui
        tab1_ui, tab2_ui = st.tabs(["📊 Results Summary", "📈 Analysis Plots"]) 
        with tab1_ui: 
            st.markdown("**Calculated Portions (Portion 1 is waste):**")
            res_data_list_ui=[]; total_calc_wt_ui=0 
            for p_item_ui in portions_disp_ui: 
                res_data_list_ui.append({"Portion #":p_item_ui.get('portion_num'),"Start Y (mm)":f"{p_item_ui.get('display_start_y'):.{DISPLAY_PRECISION}f}","End Y (mm)":f"{p_item_ui.get('display_end_y'):.{DISPLAY_PRECISION}f}","Length (mm)":f"{p_item_ui.get('length'):.{DISPLAY_PRECISION}f}","Est. Weight (g)":f"{p_item_ui.get('weight'):.{DISPLAY_PRECISION}f}"})
                total_calc_wt_ui+=p_item_ui.get('weight',0.0)
            res_df_tab_ui = pd.DataFrame(res_data_list_ui) 
            col_table_ui, col_metrics_ui_tab = st.columns([3,1.5]) 
            with col_table_ui:
                 if not res_df_tab_ui.empty:
                     st.dataframe(res_df_tab_ui.style.apply(highlight_style_func_ui,axis=None),use_container_width=True,hide_index=True)
                     btn_cols = st.columns(2)
                     with btn_cols[0]:
                         csv_ui=res_df_tab_ui.to_csv(index=False).encode('utf-8') 
                         st.download_button("Download Results CSV",csv_ui,"portions_results.csv","text/csv",key='download-csv-main-widget', use_container_width=True)
                     with btn_cols[1]:
                         current_display_cloud = st.session_state.get('point_cloud_data')
                         calc_s_y = calc_res_disp_ui.get("calc_start_y")
                         calc_e_y = calc_res_disp_ui.get("calc_end_y")
                         y_offset = calc_res_disp_ui.get("y_offset_for_plot", 0.0) 
                         if st.button("👁️ View Cuts in 3D (Open3D)", key="o3d_view_cuts_btn", use_container_width=True, disabled=not _open3d_installed or current_display_cloud is None):
                             if current_display_cloud is not None and not current_display_cloud.empty:
                                 launch_o3d_viewer_with_cuts(current_display_cloud, portions_disp_ui, calc_s_y, calc_e_y, y_offset)
                             else:
                                 st.warning("No point cloud data available for 3D view with cuts.")
                         if not _open3d_installed:
                             st.caption("Open3D view disabled (library not found).")
                 else: st.caption("No portion data for table.")
            with col_metrics_ui_tab: 
                 st.metric("Target Weight",f"{st.session_state.target_weight:.{DISPLAY_PRECISION}f} g")
                 st.metric("Tolerance",f"± {st.session_state.weight_tolerance:.{DISPLAY_PRECISION}f} g")
                 if portions_disp_ui and portions_disp_ui[0].get('weight') is not None:
                    st.metric("Portion 1 Weight",f"{portions_disp_ui[0].get('weight',0.0):.{DISPLAY_PRECISION}f} g")
                 wt_delta_ui=total_calc_wt_ui - st.session_state.total_weight if st.session_state.total_weight > FLOAT_EPSILON else 0.0 
                 delta_col_ui="off"
                 if st.session_state.total_weight > FLOAT_EPSILON : 
                    delta_col_ui = "off" if abs(wt_delta_ui)<0.1 else ("inverse" if wt_delta_ui<0 else "normal")
                    st.metric("Total Calc. Portion Wt",f"{total_calc_wt_ui:.{DISPLAY_PRECISION}f} g",f"{wt_delta_ui:.{DISPLAY_PRECISION}f} g vs Input Total",delta_color=delta_col_ui)
                 else:
                    st.metric("Total Calc. Portion Wt",f"{total_calc_wt_ui:.{DISPLAY_PRECISION}f} g")
                 density_ui,volume_ui=calc_res_disp_ui.get('density',0),calc_res_disp_ui.get('total_volume',0) 
                 st.caption(f"Density Used: {density_ui*1e6:,.2f} kg/m³ ({density_ui*1000:.3f} g/cm³)\nTotal Est. Loaf Vol: {volume_ui/1e3:,.1f} cm³")
                 if any(abs(p_val_ui.get('weight',0)-st.session_state.target_weight)<(10**(-DISPLAY_PRECISION))/2 for p_val_ui in portions_disp_ui[1:]): 
                      st.caption(f"Note: Wts rounded. Exact target hit w/o interp unusual." if st.session_state.no_interp else "Note: Wts rounded.")
        with tab2_ui: 
             st.subheader("Calculation Profiles")
             vp_ui=calc_res_disp_ui.get("volume_profile",{}) 
             ys_ui=calc_res_disp_ui.get("sorted_y_starts",np.array([]))  
             d_ui=calc_res_disp_ui.get("density",0)  
             csy_ui=calc_res_disp_ui.get("calc_start_y")  
             cey_ui=calc_res_disp_ui.get("calc_end_y")  
             if vp_ui and ys_ui.size > 0:
                 col_p1_ui, col_p2_ui = st.columns(2)  
                 with col_p1_ui:
                     with st.spinner("Generating area plot..."):
                         st.plotly_chart(plot_area_profile(vp_ui,ys_ui,st.session_state.slice_thickness,csy_ui,cey_ui),use_container_width=True)
                 with col_p2_ui:
                     with st.spinner("Generating weight plot..."):
                         st.plotly_chart(plot_cumulative_weight(vp_ui,ys_ui,d_ui,portions_disp_ui,st.session_state.target_weight,st.session_state.weight_tolerance,csy_ui,cey_ui),use_container_width=True)
             else: st.warning("Not enough data for 2D plots.")
    elif not portions_disp_ui and "Error:" not in stat_msg_ui and "Warning:" not in stat_msg_ui:
         st.warning("Calculation ran but resulted in 0 portions. Please check parameters, trims, and point cloud quality.")
elif not calc_button_main_ui and st.session_state.get('point_cloud_data') is not None : 
     st.info("Cloud loaded. Set parameters & click 'Calculate Portions'.")

# End of Script
