Version 1.0 – Published Release

This branch contains the codebase used to generate the ownership classification outputs published in the USDA Forest Service Research Data Archive:

FIA Ownership Map Dataset
https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0045

The code in this version reflects the exact workflow used to produce the published dataset and is preserved to support reproducibility of the archived results.

This version should be considered frozen and may contain known issues or inefficiencies that were present at the time of publication. Subsequent branches introduce corrections, refactoring, and performance improvements.

Characteristics of v1.0

Implements the original ownership classification pipeline used for the RDS publication.

Includes rule-based keyword classification and statistical classification components used in the final dataset generation.

Reflects the operational scripts and processing logic used during the production run.

Maintains compatibility with the original input schema and processing environment.

Known Limitations

Some ownership classifications contain known misclassifications discovered after publication.

Code structure prioritizes operational execution over modular design.

Some NoData parcels are conditionally dropped unintentionally. 
