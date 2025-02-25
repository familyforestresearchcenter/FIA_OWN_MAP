FIA_OWN_MAP
This repository contains code for processing and functions related to generating the FIA Ownership map. It includes flowcharts, preprocessing steps, and analytical scripts to support the map creation process.

Updated: The scripts will be synced with the UMass OneDrive for easier access and execution. All directory paths are configured to match the OneDrive file structure.

Processing Environment: The script is designed to run on a configured AWS EC2 instance with 256 GB of RAM.

Overall Workflow:
Main Script: Orchestrates the execution of the entire algorithm.
Random_Forest_Classifier_Training_Script: Generates a model for classifying ownership names based on a training list derived from manually classified DMP names for the NWOS implementation. This training list contains confidential data, so it must be stored on an external flash drive. This script only needs to be run when a new training set is available. Otherwise, the classify_unknown_ownership_model.pkl file, which is the authoritative model, should be used.
Preprocessing Script: Handles basic layer joins, including Parcel data, county data, PAD data, Indigenous lands data, and land cover data. It also calculates key descriptors such as parcel areas, centroids, and summarizes land cover information within each parcel.
Classify_Unknowns Script: Applies hierarchical and machine learning classification techniques. Ownership names are first analyzed for structural components, followed by keyword searches to classify them. Remaining unknown ownerships are then classified using the random forest model created earlier.
Name_Matching Script: Aggregates ownership records for individuals who own multiple parcels across the state. Due to non-standardized reporting from county offices, ownership data may differ slightly for the same individual (e.g., "V. Harris" vs. "Vance Harris"). This script matches and consolidates such variations using phonetic algorithms to reduce the number of unique ownership records.
Summary Script: Calculates summary statistics for the forested parcels and the various ownership classes.
New_Map_Data Script: Converts the calculated tabular information into a new raster that represents the data in a continuous spatial format.
NoData Overlay: Due to the nature of the sourced parcel data, some raster cells may contain NoData values. This final overlay fills in these gaps with Public Ownership data from the PAD layer and Indigenous Lands data from the Indigenous_Lands_BIA_AIAN_National_LAR layer.
