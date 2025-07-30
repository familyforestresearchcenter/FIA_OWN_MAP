
FIA_OWN_MAP
This repository contains a set of Python scripts for generating the FIA Ownership Map, a spatial dataset that combines parcel-level ownership data, land cover information, and overlays from public land datasets. The repository provides a full data-processing pipeline, from raw parcel data to a final raster product and summary tables.

Key Features
Automated Preprocessing: Integrates parcel polygons with land cover data, county boundaries, and Indigenous Lands.

Ownership Classification: Uses both rule-based keyword detection and a trained Random Forest model to classify ownership types.

Name Matching: Consolidates ownership records using phonetic matching (Double Metaphone) to unify slight variations in owner names.

Raster Generation: Produces a final encoded raster combining ownership classes and NLCD land cover data.

Overlay Corrections: Applies PADUS (Protected Areas Database of the U.S.) and Indigenous Lands overlays to fill in missing or misclassified areas.

Processing Environment
Recommended Hardware: The pipeline is optimized for a 256 GB RAM AWS EC2 instance due to the size of the geospatial datasets.

File System: Paths are configured to match the UMass OneDrive directory structure for data synchronization and ease of execution.

Dependencies: Requires Python 3.9+ with libraries including geopandas, rasterio, xarray, scikit-learn, numpy, pandas, and others.

Overall Workflow
Main Script (Main.py)
Orchestrates the execution of all pipeline steps.

Cleans up intermediate files and packages final outputs into a {STATE}.zip file containing:

{STATE}_Full_Data_Table.csv

{STATE}_Final_Encoded.tif

{STATE}_Parcel_IDs.json

Step-by-Step Modules
1. Preprocessing (Preprocessing_opt.py)
Reads parcel data from .gdb files and converts polygons to raster.

Clips NLCD 10m raster to the parcel extent.

Calculates land cover proportions per parcel.

Joins county boundaries and Indigenous Lands (ILs).

Computes parcel geometry (centroid coordinates, area).

Outputs:

temp.csv – Intermediate table with land cover stats and parcel metadata.

{STATE}_Parcel_IDs.json – GeoJSON of parcel IDs and geometries.

2. Ownership Classification (Classify_Unknowns_opt.py)
Cleans and normalizes ownership names (e.g., removing punctuation, handling Unicode).

Performs hierarchical classification:

Keyword-based detection: Detects Federal, State, Local government, corporations, trusts, religious groups, family ownerships.

Machine learning model: Remaining unknowns are classified using a Random Forest model (classify_unknown_ownership_model.pkl) trained on a curated dataset of ownership names.

Outputs:

new_classified_state_temp.csv – Classified ownership data.

3. Name Matching (Name_Matching_opt.py)
Groups parcels owned by the same entity despite name variations:

Uses Double Metaphone phonetic encoding.

Considers address combinations for disambiguation.

Assigns a Unq_ID (unique ownership identifier) for each owner.

Outputs:

Full_Data_Table.csv – Consolidated ownership data with parcel counts per owner.

4. Summary (Summary_Script_opt.py)
Computes forest area and parcel statistics:

Forest_Area = forested land in acres per parcel.

Counts forested parcels and aggregates totals by owner.

Standardizes column names to FIA-compatible codes (e.g., OWNCD, NLCD_41_PROP).

Outputs:

Updated Full_Data_Table.csv.

5. Map Data Encoding (Map_Data_opt.py)
Joins ownership codes (OWNCD) with parcel geometry.

Combines ownership and NLCD data into a single encoded raster using a reclassification dictionary (New_Raster_Reclass.pickle).

Outputs:

Own_Type_by_Landcover_Encoded.tif.

6. Final Overlay (Last_Overlay_opt.py)
Applies PADUS (Protected Areas Database of the US) and Indigenous Lands overlays to fill NoData gaps or refine classifications.

Outputs:

{STATE}_Final_Encoded.tif – The final raster dataset.

Random Forest Classifier
The Random_Forest_Classifier_Training_Script trains a model for classifying ownership names.

The training data (derived from NWOS implementation) is confidential and stored on an external flash drive.

Normally, the pipeline uses the pre-trained classify_unknown_ownership_model.pkl.

Usage
Place the input .gdb file for the state into /dev/shm/.

Run:

bash
Copy
Edit
python Main.py
The final outputs will be written to /dev/shm/{STATE}.zip.
