**Point Cloud Portioning & Slicing Software.**
```
Includes Rockwell & Siemens PLC industrial protocol read, write & error handling. 
Mitsubishi support coming soon.
```

This repository contains two-part automation for calculating optimal cutting portions from 3D point cloud data. The system is designed for industrial protocol applications where a "headless" processing engine is triggered by new scan files, and a separate, user-friendly web interface is used for viewing and analyzing the results.

Dashboard preview

<img width="1918" height="1074" alt="image" src="https://github.com/user-attachments/assets/91f28e8d-7203-4d23-809f-d02253f99a16" />

Console log preview

![image](https://github.com/user-attachments/assets/f67a7674-f821-452f-b3dc-b918b4905c26)

**Core Features**

**- Headless Processing Engine (run_pipeline.py):** Command-line script that can run as a continuous "watcher" service, automatically processing new point cloud files as they arrive.

**- File Support:** Natively handles a wide range of formats (.xyz, .pcd, .ply, .csv, .xlsx, .xls).

**- Pre-Processing:** Includes a pipeline of optional steps:

    - PCA Alignment: Corrects for skewed or rotated scans.
    - Auto Downsampling: Manages large point clouds by reducing point count to a set threshold.
    - Radius Outlier Removal (ROR): Cleans noisy data with an auto-estimation mode for the radius parameter.
  
**- Calculation:** 

    - Calculates portion weights based on volumetric slicing, with support for trims, blade kerf, and interpolation etc. 
    - "Convex Hull" OR "Alpha Shape" algorithms available for volumetric slicing. 

**- PLC Integration (pylogix):**

    - Reads processing parameters (e.g., target weight, product weight etc..) directly from Rockwell PLCs (ControlLogix/CompactLogix).
    - Writes key results (portion counts, yield, waste, heartbeat, status) back to the PLC for a closed-loop system.

**- Data Archiving & Cleanup:** Automatically saves each run's complete data payload to a timestamped archive and cleans up old archives to manage disk space.

**- Streamlit Results Viewer (resultsdashboard.py):** A separate, web-based dashboard for analysis.

    - Dashboard: Displays metrics like Yield, Waste, and Product Length.
    - Recent Runs Browser: load and compare historical runs from the archive.
    - Interactive 3D Viewers: Includes both a static in-app plot and buttons to launch an interactive Open3D window to inspect the point cloud and calculated cut planes.
    - Slice Inspector: An interactive tool to view the 2D cross-section and calculated shape of any slice along the product.
    
**- functionlib.py (The Library):** This file contains all the core, reusable functions for point cloud processing, calculations, plotting, and file loading. It does not contain any application logic itself.

**- run_pipeline.py (The Headless Engine):** This is the primary processing script, designed to be run from the command line or as an automated service.
It runs in "watcher" mode, continuously scanning an input directory for new point cloud files.

    - When a file is found, it executes the entire processing pipeline based on parameters defined in its configuration section.
    - It saves a detailed output file (.pkl) containing all results, logs, and processed data.
    - It communicates with a PLC (if enabled) to read parameters and write back results.
    - resultsdashboard.py (The Streamlit Viewer):
    - A completely separate web application for visualization. It does not perform any calculations. Its only job is to load the .pkl payload files generated by the headless engine. 
    - It provides a user-friendly interface to browse recent runs, view 3D models, analyze results tables, and inspect slice profiles.
    
**Setup and Installation**

**1. Prerequisites**
```
Python 3.8+
A Rockwell PLC on the network (if using PLC mode)
```

**2. Clone the Repository**
```
  git clone <your-repository-url>
  cd <your-repository-name>
```

**3. Install Required Python Libraries**
**It is highly recommended to use a virtual environment.**
```
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**Install all required libraries**
```
pip install numpy pandas streamlit open3d pylogix joblib openpyxl scikit-learn alphashape shapely
```

**4. File Structure**
```
Ensure your project directory is set up as follows:
project_folder/
├── functionlib.py               # The main library of functions
├── run_pipeline.py              # The headless processing engine
├── resultsdashboard.py              # The Streamlit viewer app
|
├── xyz_input/                   # Folder where new scan files are placed
│
├── xyz_processed/               # Folder where successfully processed scans are moved
│   └── error/                   # Subfolder for scans that failed processing
│
├── run_archives/                # Folder where all result payloads are archived
│   └── payload_YYYY-MM-DD_HH-MM-SS.pkl
|
└── latest_run_payload.pkl       # A copy of the most recent result payload for easy loading
```

**Step 1: Configure the Headless Engine**
```
Open run_pipeline.py and edit the CONFIGURATION SECTION at the top.

ARCHIVE_...: Configure archiving and cleanup settings.

PLC_MODE: Set to True or False to enable/disable PLC communication.

PLC_IP_ADDRESS: Enter your PLC's IP address.

PLC_TAG_MAPPING / PLC_WRITE_TAG_MAPPING: Crucially, update these dictionaries with the exact tag names from your PLC program.

PIPELINE_PARAMS: Set the default processing parameters. These will be used if PLC_MODE is False or if a specific tag cannot be read from the PLC.
```
**Step 2: Run the Headless Engine**
```
Open a terminal, navigate to the project directory, and start the watcher service:
python run_pipeline.py
The script will now be running, waiting for files to appear in the xyz_input folder.
Auto run batch process could be used to auto start the application.
```
**Step 3: Trigger a Process**
```
To trigger a run, move a supported scan file (e.g., scanner_data.xyz) into the xyz_input folder. The watcher service will automatically detect it, process it, and save the results.
Once it has completed the pipeline process it will automatically be moved to xyz_processed folder.
```
**Step 4: View the Results**
```
Open a second, separate terminal.
Navigate to the same project directory.
Launch the Streamlit viewer app:
streamlit run resultsdashboard.py
Auto run batch process could be used to auto start the application.
Your web browser will open to the dashboard.
In the sidebar, click "Load Latest Run" to see the results from the file you just processed, or use the "Recent Runs" dropdown to select a specific archived run.
```
