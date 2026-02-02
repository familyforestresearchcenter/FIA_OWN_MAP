import time
from datetime import timedelta
import csv
from collections import defaultdict

import numpy as np
import rasterio as rio

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
OWNERSHIP_RASTER = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\US_Ownership_Landcover_confidence_v1_q100.tif"
NLCD_RASTER      = r"F:\OwnershipMap\New_Script\Error_Calculations\Annual_NLCD_LndCnf_2020_ALIGNED_TO_OWNERSHIP.tif"

OUTPUT_CSV = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\confidence_histogram_comparison.csv"

# ------------------------------------------------------------
# Init
# ------------------------------------------------------------
t0 = time.time()

hist_own = defaultdict(np.uint64)
hist_nlcd = defaultdict(np.uint64)

own_min = None
own_max = None
nlcd_min = None
nlcd_max = None

block_count = 0
total_pixels = 0

# ------------------------------------------------------------
# Open both rasters (alignment assumed)
# ------------------------------------------------------------
with rio.open(OWNERSHIP_RASTER) as own_src, rio.open(NLCD_RASTER) as nlcd_src:

    # Safety checks
    if own_src.transform != nlcd_src.transform:
        raise ValueError("Transforms do not match")
    if own_src.crs != nlcd_src.crs:
        raise ValueError("CRS do not match")
    if own_src.width != nlcd_src.width or own_src.height != nlcd_src.height:
        raise ValueError("Dimensions do not match")

    print("Ownership dtype:", own_src.dtypes[0])
    print("NLCD dtype:", nlcd_src.dtypes[0])

    # --------------------------------------------------------
    # Crawl by native block windows
    # --------------------------------------------------------
    for (_, window) in own_src.block_windows(1):

        own_data = own_src.read(1, window=window)
        nlcd_data = nlcd_src.read(1, window=window)

        # Flatten
        own_flat = own_data.ravel()
        nlcd_flat = nlcd_data.ravel()

        # Mask nodata independently
        if own_src.nodata is not None:
            own_flat = own_flat[own_flat != own_src.nodata]
        if nlcd_src.nodata is not None:
            nlcd_flat = nlcd_flat[nlcd_flat != nlcd_src.nodata]

        if own_flat.size == 0:
            continue

        # Update min/max
        o_min = own_flat.min()
        o_max = own_flat.max()
        n_min = nlcd_flat.min()
        n_max = nlcd_flat.max()

        own_min = o_min if own_min is None else min(own_min, o_min)
        own_max = o_max if own_max is None else max(own_max, o_max)
        nlcd_min = n_min if nlcd_min is None else min(nlcd_min, n_min)
        nlcd_max = n_max if nlcd_max is None else max(nlcd_max, n_max)

        # Histogram via exact value counts
        vals, counts = np.unique(own_flat, return_counts=True)
        for v, c in zip(vals, counts):
            hist_own[int(v)] += int(c)

        vals, counts = np.unique(nlcd_flat, return_counts=True)
        for v, c in zip(vals, counts):
            hist_nlcd[int(v)] += int(c)

        total_pixels += own_flat.size
        block_count += 1

        if block_count % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"blocks={block_count:,}  "
                f"pixels={total_pixels:,}  "
                f"own_min={own_min} own_max={own_max}  "
                f"nlcd_min={nlcd_min} nlcd_max={nlcd_max}  "
                f"elapsed={timedelta(seconds=int(elapsed))}"
            )

# ------------------------------------------------------------
# Write CSV
# ------------------------------------------------------------
elapsed = time.time() - t0
print("\n📤 Writing histogram comparison CSV...")

all_values = sorted(set(hist_own.keys()) | set(hist_nlcd.keys()))

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "value",
        "ownership_pixel_count",
        "ownership_percent",
        "nlcd_pixel_count",
        "nlcd_percent"
    ])

    for v in all_values:
        own_count = hist_own.get(v, 0)
        nlcd_count = hist_nlcd.get(v, 0)

        own_pct = (own_count / total_pixels * 100) if total_pixels else 0
        nlcd_pct = (nlcd_count / total_pixels * 100) if total_pixels else 0

        writer.writerow([
            v,
            int(own_count),
            round(own_pct, 6),
            int(nlcd_count),
            round(nlcd_pct, 6)
        ])

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------
print("\n=== FINAL SUMMARY ===")
print("Total pixels:", f"{total_pixels:,}")
print("Ownership min/max:", own_min, own_max)
print("NLCD min/max:", nlcd_min, nlcd_max)
print("Elapsed:", timedelta(seconds=int(elapsed)))
print(f"\n✅ Histogram written to:\n{OUTPUT_CSV}")
