# FIA_OWN_MAP

This repository contains Python scripts for processing and generating the **FIA Ownership Map**, a spatial dataset combining parcel-level ownership data, land cover information, and overlays from public land datasets. The pipeline integrates preprocessing, ownership classification, name matching, and rasterization steps to create a final, encoded raster product along with detailed tabular outputs.

---

## Key Features
- **Automated Preprocessing:** Integrates parcel polygons with land cover, county boundaries, PADUS data, and Indigenous Lands.
- **Ownership Classification:** Uses rule-based keyword detection and a pre-trained **Random Forest model** to classify ownership types.
- **Name Matching:** Consolidates ownership records using **phonetic algorithms (Double Metaphone)** to unify variations in owner names.
- **Raster Generation:** Produces a raster encoding both ownership classes and NLCD land cover data.
- **Overlay Adjustments:** Applies PADUS (Protected Areas Database of the U.S.) and Indigenous Lands overlays to fill or refine classifications.

---

## Processing Environment
- **Hardware Requirements:** Optimized for an **AWS EC2 instance with 256 GB RAM** due to large geospatial datasets.
- **File System:** Paths are configured to match the **UMass OneDrive** directory structure for synchronized data and scripts.
- **Dependencies:** Python 3.9+ with libraries such as:
  - `geopandas`, `rasterio`, `xarray`, `rioxarray`
  - `numpy`, `pandas`, `tqdm`, `shapely`
  - `scikit-learn`, `metaphone`
  - `numpy_indexed`, `geocube`

---

## Overall Workflow

### **Main Script (`Main.py`)**
`Main.py` orchestrates the full pipeline. It sequentially runs all modules, cleans up intermediate files, and packages final outputs into a `{STATE}.zip` file containing:
- `{STATE}_Full_Data_Table.csv`
- `{STATE}_Final_Encoded.tif`
- `{STATE}_Parcel_IDs.json`

---

### **Step-by-Step Modules**

#### **1. Preprocessing (`Preprocessing_opt.py`)**
- Reads parcel data from a `.gdb` file.
- Converts parcels to raster.
- Clips **NLCD 10m raster** to parcel extent.
- Calculates **land cover proportions** for each parcel.
- Joins **county boundaries** and **Indigenous Lands (ILs)**.
- Computes parcel descriptors: centroid coordinates and area.
- **Outputs:**
  - `temp.csv` – Parcel data with land cover stats and county info.
  - `{STATE}_Parcel_IDs.json` – GeoJSON of parcel geometries and IDs.

#### **2. Ownership Classification (`Classify_Unknowns_opt.py`)**
- Normalizes and cleans ownership names (`OWN1`, `OWN2`).
- **Keyword classification:** Detects Federal, State, Local government, corporate, religious, trust, and family ownerships.
- **Machine learning classification:** Remaining unknowns are classified using a **Random Forest model** (`classify_unknown_ownership_model.pkl`) trained on manually labeled NWOS data.
- **Outputs:**
  - `new_classified_state_temp.csv` – Ownership classifications.

#### **3. Name Matching (`Name_Matching_opt.py`)**
- Groups parcels by ownership using:
  - **Double Metaphone** phonetic encoding.
  - Address-based grouping to resolve variations.
- Assigns a **`Unq_ID`** (unique ownership identifier).
- **Outputs:**
  - `Full_Data_Table.csv` – Consolidated ownership records with parcel counts.

#### **4. Summary (`Summary_Script_opt.py`)**
- Calculates:
  - **Forest area per parcel (acres).**
  - **Forest parcel counts per owner.**
- Aggregates total forested area and parcel statistics by ownership ID.
- Standardizes column names to FIA codes (e.g., `OWNCD`, `NLCD_41_PROP`).
- **Outputs:**
  - Updated `Full_Data_Table.csv` with forest metrics.

#### **5. Map Data Encoding (`Map_Data_opt.py`)**
- Joins ownership codes (`OWNCD`) to parcels.
- Combines ownership codes and NLCD data into an **encoded raster**.
- Applies a **reclassification dictionary** (`New_Raster_Reclass.pickle`).
- **Outputs:**
  - `Own_Type_by_Landcover_Encoded.tif`.

#### **6. Final Overlay (`Last_Overlay_opt.py`)**
- Applies **PADUS** and **Indigenous Lands** overlays to refine or fill gaps in ownership coding.
- **Outputs:**
  - `{STATE}_Final_Encoded.tif`.

---

## Random Forest Classifier
- The **Random_Forest_Classifier_Training_Script** trains the ownership classification model.
- Training data (from NWOS) is **confidential** and stored externally.
- The pipeline typically uses the pre-trained model:  
  `classify_unknown_ownership_model.pkl`.

---

## Usage
1. Place the input `.gdb` file for the state into `/dev/shm/`.
2. Run the main pipeline:
   ```bash
   python Main.py

## Final Raster Mosaicking (ArcGIS Pro)

The final step in the FIA Ownership Map workflow is performed in **ArcGIS Pro** and produces a seamless national raster by:

1. **Clipping each state raster** to its corresponding state boundary using the 2019 Census state boundaries hosted on ArcGIS Online (AGOL):  
   *BND - States 500K (Census 2019)*  
   Source: [Esri Feature Service](https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/BND___States_500K__Census_2019_/FeatureServer)

2. **Mosaicking the clipped rasters** using the `Mosaic To New Raster` tool in ArcGIS Pro to create a contiguous, nationwide raster surface that encodes ownership type by land cover.

This step ensures all outputs are cleanly clipped, standardized, and aligned to authoritative boundaries for downstream visualization and spatial analysis.

