# convert_cloud.py
# -----------------------------------------------------------------------------
# Description: A command-line utility to convert point cloud files from one
#              format to another. It supports reading any format handled by
#              functionlib and writing to common formats like PCD and PLY.
#
# Usage:
#   python convert_cloud.py <input_file> <output_file>
#
# Examples:
#   python convert_cloud.py my_scan.xyz converted_scan.pcd
#   python convert_cloud.py my_data.csv converted_scan.ply
#   python convert_cloud.py my_scan.xyz converted_scan.pcd --ascii (output to be saved in ASCII format, can reduce Open3D read speed but makes the file readable)
# -----------------------------------------------------------------------------

import sys
import os
import argparse # Used for handling command-line arguments

# --- Import necessary functions from your library ---
try:
    from functionlib import (
        load_point_cloud_from_file,
        _open3d_installed
    )
    if _open3d_installed:
        import open3d as o3d
    else:
        # If open3d is not installed, we can't write PCD/PLY files.
        print("ERROR: Open3D library is required for this script to write .pcd or .ply files.")
        print("       Please install it with: pip install open3d")
        sys.exit(1)

except ImportError as e:
    print(f"FATAL ERROR: Could not import from 'FunctionLib.py': {e}")
    print("             Please ensure it is in the same directory.")
    sys.exit(1)

def main():
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Convert point cloud files to different formats.",
        epilog="Example: python convert_cloud.py input.xyz output.pcd"
    )
    parser.add_argument("input_file", help="Path to the input point cloud file.")
    parser.add_argument("output_file", help="Path for the converted output file (e.g., output.pcd, output.ply).")
    parser.add_argument(
        "--ascii", 
        action="store_true", 
        help="If specified, saves the output file in ASCII (text) format. Default is binary."
    )
    
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    write_ascii = args.ascii

    print(f"--- Point Cloud Converter ---")
    
    # --- 1. Load the input file ---
    print(f"Loading input file: {input_path}")
    points_numpy = load_point_cloud_from_file(input_path)

    if points_numpy is None or points_numpy.shape[0] == 0:
        print("\nConversion failed: Could not load valid data from the input file.")
        sys.exit(1)
    
    print(f"Successfully loaded {len(points_numpy)} points.")

    # --- 2. Convert to Open3D format ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_numpy)

    # --- 3. Save to the specified output format ---
    print(f"\nSaving converted file to: {output_path}")
    print(f"Format: {'ASCII (Text)' if write_ascii else 'Binary'}")

    try:
        # Open3D's write_point_cloud function handles the format based on the file extension.
        success = o3d.io.write_point_cloud(output_path, pcd, write_ascii=write_ascii)
        if not success:
            raise IOError("Open3D reported a failure to write the file.")
        
        print("\n✅ Conversion successful!")
        print(f"File saved at: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"\n❌ Conversion failed: An error occurred during saving.")
        print(f"   Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
