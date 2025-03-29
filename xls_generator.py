import pandas as pd
import numpy as np
import random

# --- Configuration (same as PCD generator for consistency) ---
NOMINAL_LENGTH = 360.0 # mm (Y-axis)
NOMINAL_WIDTH = 90.0   # mm (X-axis)
NOMINAL_HEIGHT = 90.0  # mm (Z-axis)
DENSITY_FACTOR = 0.5   # Adjust for point density (lower = more points)
NOISE_STDDEV = 0.3     # mm
DIMENSION_VARIATION = 0.05
WAVINESS_FACTOR_X = 0.04
WAVINESS_FACTOR_Z = 0.06
WAVINESS_FREQ_MIN = 1.5
WAVINESS_FREQ_MAX = 3.5

# py xls.py - Generate default nominal loaf (full 3D) to cheese_loaf_test.xlsx
# py xls.py -o deformed_ts.xlsx --deformed --scan_type top_sides - Generate deformed loaf (top & sides scan) to deformed_ts.xlsx
# py xls.py -o nominal_5k.xlsx --num_points 5000 - Generate nominal loaf with exactly 5000 points

# --- Helper Functions (Copied from PCD generator) ---
def generate_surface_points(length, width, height, density_factor, faces):
    """Generates random points on the specified faces of a cuboid."""
    points = []
    total_area = 0
    face_areas = {'top': width*length, 'bottom': width*length, 'front': height*length, 'back': height*length, 'left': width*height, 'right': width*height}
    for face in faces: total_area += face_areas.get(face, 0)
    if total_area == 0: return np.empty((0, 3))
    points_per_mm2 = 1.0 / (density_factor**2) if density_factor > 0 else 1.0
    total_points_target = int(total_area * points_per_mm2)
    # Generate points for requested faces proportionally... (same logic as before)
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
    if 'right' in faces: # Max Y
        n = max(10, int(total_points_target * (face_areas['right'] / total_area)))
        x = np.random.uniform(0, width, n); z = np.random.uniform(0, height, n)
        points.append(np.column_stack([x, np.full_like(x, length), z]))
    if 'left' in faces: # Min Y
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
    import argparse # Use argparse for command-line options

    parser = argparse.ArgumentParser(description="Generate a synthetic cheese loaf point cloud Excel file (.xlsx).")
    parser.add_argument("-o", "--output", default="cheese_loaf_test.xlsx", help="Output Excel filename.")
    parser.add_argument("--deformed", action="store_true", help="Apply deformation (waviness, dimension variation).")
    parser.add_argument("--scan_type", choices=['full', 'top', 'top_sides'], default='full',
                        help="'full': All faces. 'top': Top only. 'top_sides': Top + 4 sides.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num_points", type=int, default=None, help="Optional: Specify exact number of points instead of using density.")

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

    # Determine faces
    if args.scan_type == 'top': faces_to_generate = ('top',)
    elif args.scan_type == 'top_sides': faces_to_generate = ('top', 'front', 'back', 'left', 'right')
    else: faces_to_generate = ('top', 'bottom', 'front', 'back', 'left', 'right')
    print(f"Scan Type: {args.scan_type.replace('_',' ').title()} (generating: {', '.join(faces_to_generate)})")

    # Generate Base Points
    print(f"Generating base points...")
    points = generate_surface_points(length, width, height, DENSITY_FACTOR, faces=faces_to_generate)

    # Adjust number of points if specified
    if args.num_points is not None:
        if args.num_points > 0 and points.shape[0] > 0:
             print(f"Adjusting number of points from {points.shape[0]} to {args.num_points}...")
             indices = np.random.choice(points.shape[0], args.num_points, replace=(args.num_points > points.shape[0]))
             points = points[indices]
        elif args.num_points <= 0:
             print("Number of points set to 0, generating empty file.")
             points = np.empty((0, 3))

    print(f"Generated {points.shape[0]} base points.")

    # Apply Deformation
    if args.deformed: print("Applying deformation..."); points = apply_deformation(points, length, width, height)

    # Add Noise
    print(f"Adding noise (Std Dev: {NOISE_STDDEV} mm)..."); points = add_noise(points, NOISE_STDDEV)

    # Clip Z and Center Y
    min_z_clip = -2.0 * NOISE_STDDEV if 'bottom' in faces_to_generate else -0.5
    points[:, 2] = np.clip(points[:, 2], min_z_clip, None)
    points[:, 1] = np.clip(points[:, 1], -length * 0.05, length * 1.05)
    if points.shape[0] > 0: points[:, 1] -= np.min(points[:, 1])

    # --- Create Pandas DataFrame ---
    # Explicitly name columns 'x', 'y', 'z' for compatibility with the main app
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    print(f"Final point count: {len(df)}")

    # --- Save to Excel ---
    print(f"Saving data to '{args.output}'...")
    try:
        # Use 'openpyxl' engine for .xlsx
        df.to_excel(args.output, index=False, sheet_name='PointCloudData', engine='openpyxl')
        print("Save successful.")
    except ImportError:
         print("\nError: Could not save Excel file.")
         print("The 'openpyxl' library is required. Please install it:")
         print("  pip install openpyxl")
    except Exception as e:
        print(f"Error saving Excel file: {e}")