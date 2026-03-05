Version 1.1 – Corrected Analysis Release

This branch represents a minimally updated version of v1.0, incorporating several corrections identified after the original publication.

While the overall structure and classification workflow remain largely unchanged from the published pipeline, this version fixes issues discovered during subsequent validation and analysis.

Version 1.1 is the codebase used to generate the SQLite analysis database derived from the ownership classification outputs.

Key Changes from v1.0

Fixed errors related to dropping parcels with NoData values during preprocessing.

Corrected several classification behaviors identified during post-publication review.

Maintains the same overall pipeline and architecture used in the published workflow.

Relationship to v1.0

Core logic and structure remain nearly identical to the published pipeline.

Changes are limited to targeted bug fixes and corrections rather than architectural changes.

Results are largely consistent with v1.0 but remove known processing errors.

Intended Use

This version should be used when working with the SQLite database analysis outputs derived from the ownership dataset.

Later versions of the repository introduce more substantial refactoring and improvements to the classification system.
