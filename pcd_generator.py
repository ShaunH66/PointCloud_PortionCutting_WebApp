import numpy as np
import open3d as o3d
import argparse
import random

# --- Configuration ---
NOMINAL_LENGTH = 360.0 # mm (Y-axis)
NOMINAL_WIDTH = 90.0   # mm (X-axis)
NOMINAL_HEIGHT = 90.0  # mm (Z-axis)
DENSITY_FACTOR = 0.5
NOISE_STDDEV = 0.3
DIMENSION_VARIATION = 0.05
WAVINESS_FACTOR_X = 0.04
WAVINESS_FACTOR_Z = 0.06
WAVINESS_FREQ_MIN = 1.5
WAVINESS_FREQ_MAX = 3.5

# python pcd.py -o nominal_full.pcd --visualize
# python pcd.py -o deformed_full.pcd --deformed --visualize

# python pcd.py -o nominal_top.pcd --scan_type top --visualize
# python pcd.py -o deformed_top.pcd --deformed --scan_type top --visualize

# python pcd.py -o nominal_top_sides.pcd --scan_type top_sides --visualize
# python pcd.py -o deformed_top_sides.pcd --deformed --scan_type top_sides --visualize

# --- Helper Functions ---
def generate_surface_points(length, width, height, density_factor, faces=('top', 'bottom', 'front', 'back', 'left', 'right')):
    """Generates random points on the specified faces of a cuboid."""
    points = []
    total_area = 0
    face_areas = {
        'top': width * length, 'bottom': width * length,
        'front': height * length, 'back': height * length,
        'left': width * height, 'right': width * height
    }

    for face in faces: total_area += face_areas.get(face, 0)
    if total_area == 0: return np.empty((0, 3))

    points_per_mm2 = 1.0 / (density_factor**2) if density_factor > 0 else 1.0
    total_points_target = int(total_area * points_per_mm2)

    if 'top' in faces:
        n = max(10, int(total_points_target * (face_areas['top'] / total_area)))
        x = np.random.uniform(0, width, n); y = np.random.uniform(0, length, n)
        points.append(np.column_stack([x, y, np.full_like(x, height)]))
    if 'bottom' in faces:
        n = max(10, int(total_points_target * (face_areas['bottom'] / total_area)))
        x = np.random.uniform(0, width, n); y = np.random.uniform(0, length, n)
        points.append(np.column_stack([x, y, np.zeros_like(x)]))
    if 'front' in faces: # Max X
        n = max(10, int(total_points_target * (face_areas['front'] / total_area)))
        y = np.random.uniform(0, length, n); z = np.random.uniform(0, height, n)
        points.append(np.column_stack([np.full_like(y, width), y, z]))
    if 'back' in faces: # Min X
        n = max(10, int(total_points_target * (face_areas['back'] / total_area)))
        y = np.random.uniform(0, length, n); z = np.random.uniform(0, height, n)
        points.append(np.column_stack([np.zeros_like(y), y, z]))
    if 'right' in faces: # Max Y (End face)
        n = max(10, int(total_points_target * (face_areas['right'] / total_area)))
        x = np.random.uniform(0, width, n); z = np.random.uniform(0, height, n)
        points.append(np.column_stack([x, np.full_like(x, length), z]))
    if 'left' in faces: # Min Y (Start face)
        n = max(10, int(total_points_target * (face_areas['left'] / total_area)))
        x = np.random.uniform(0, width, n); z = np.random.uniform(0, height, n)
        points.append(np.column_stack([x, np.zeros_like(x), z]))

    if not points: return np.empty((0, 3))
    return np.concatenate(points, axis=0)

def apply_deformation(points, length, width, height):
    """Applies waviness/bulges to the points."""
    if points.shape[0] == 0: return points
    freq_x = random.uniform(WAVINESS_FREQ_MIN, WAVINESS_FREQ_MAX)
    freq_z = random.uniform(WAVINESS_FREQ_MIN, WAVINESS_FREQ_MAX)
    amp_x = width * WAVINESS_FACTOR_X; amp_z = height * WAVINESS_FACTOR_Z
    norm_y = points[:, 1] / length if length > 0 else points[:, 1] * 0
    delta_x = amp_x * np.sin(norm_y * freq_x * 2 * np.pi)
    delta_z = amp_z * np.cos(norm_y * freq_z * 2 * np.pi)
    points[:, 0] += delta_x; points[:, 2] += delta_z
    return points

def add_noise(points, noise_stddev):
    """Adds Gaussian noise to points."""
    if points.shape[0] == 0: return points
    return points + np.random.normal(0, noise_stddev, points.shape)

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic cheese loaf point cloud (PCD).")
    parser.add_argument("-o", "--output", default="cheese_loaf.pcd", help="Output PCD filename.")
    parser.add_argument("--deformed", action="store_true", help="Apply deformation (waviness, dimension variation).")
    # --- Updated choices ---
    parser.add_argument("--scan_type", choices=['full', 'top', 'top_sides'], default='full',
                        help="'full': All faces. 'top': Top face only. 'top_sides': Top and 4 vertical sides/ends.")
    parser.add_argument("--visualize", action="store_true", help="Show the generated point cloud using open3d.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Determine dimensions
    length = NOMINAL_LENGTH; width = NOMINAL_WIDTH; height = NOMINAL_HEIGHT
    if args.deformed:
        length *= (1 + random.uniform(-DIMENSION_VARIATION, DIMENSION_VARIATION))
        width *= (1 + random.uniform(-DIMENSION_VARIATION, DIMENSION_VARIATION))
        height *= (1 + random.uniform(-DIMENSION_VARIATION, DIMENSION_VARIATION))
        length, width, height = max(1, length), max(1, width), max(1, height)
        print(f"Applying Deformation: Dimensions ~ L={length:.1f}, W={width:.1f}, H={height:.1f}")
    else:
        print(f"Using Nominal Dimensions: L={length:.1f}, W={width:.1f}, H={height:.1f}")

    # --- Determine faces based on scan type ---
    if args.scan_type == 'top':
        faces_to_generate = ('top',)
        print("Scan Type: Simulating Top-Down (generating top face only)")
    elif args.scan_type == 'top_sides': # <-- New Option
        faces_to_generate = ('top', 'front', 'back', 'left', 'right')
        print("Scan Type: Simulating Top & Sides (generating top and 4 vertical faces)")
    else: # 'full' scan
        faces_to_generate = ('top', 'bottom', 'front', 'back', 'left', 'right')
        print("Scan Type: Simulating Full 3D (generating all faces)")

    # Generate Base Points
    print(f"Generating base points (Density Factor: {DENSITY_FACTOR})...")
    points = generate_surface_points(length, width, height, DENSITY_FACTOR, faces=faces_to_generate)
    print(f"Generated {points.shape[0]} base points.")

    # Apply Deformation (if requested)
    if args.deformed:
        print("Applying deformation..."); points = apply_deformation(points, length, width, height)

    # Add Noise
    print(f"Adding noise (Std Dev: {NOISE_STDDEV} mm)..."); points = add_noise(points, NOISE_STDDEV)

    # Clip Z coordinates
    # Allow small negative Z only if the bottom was explicitly generated ('full' scan)
    min_z_clip = -2.0 * NOISE_STDDEV if 'bottom' in faces_to_generate else -0.5 # Allow small dip below 0 otherwise
    points[:, 2] = np.clip(points[:, 2], min_z_clip, None)

    # Clip Y and Center near 0
    points[:, 1] = np.clip(points[:, 1], -length * 0.05, length * 1.05)
    if points.shape[0] > 0: points[:, 1] -= np.min(points[:, 1])

    # Create Open3D Point Cloud
    if points.shape[0] == 0:
        print("Warning: No points generated.")
    else:
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
        print(f"Final point count: {len(pcd.points)}")

        # Save PCD File
        print(f"Saving point cloud to '{args.output}'...");
        try:
            o3d.io.write_point_cloud(args.output, pcd, write_ascii=True); print("Save successful.")
        except Exception as e: print(f"Error saving PCD file: {e}")

        # Visualize
        if args.visualize:
            print("Visualizing point cloud... Close the window to exit.")
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(width, height) * 0.5)
            o3d.visualization.draw_geometries([pcd, coord_frame])