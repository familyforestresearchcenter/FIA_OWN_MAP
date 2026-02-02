import time
from datetime import timedelta

import numpy as np
import rasterio as rio

# RASTER = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\US_Ownership_Landcover_confidence_v1_q100.tif"
RASTER = r"F:\OwnershipMap\New_Script\Error_Calculations\Annual_NLCD_LndCnf_2020_ALIGNED_TO_OWNERSHIP.tif"



t0 = time.time()
gmin = 255
gmax = 0
dtype = None
count = 0

with rio.open(RASTER) as src:
    dtype = src.dtypes[0]
    # iterate using the raster's native blocks (fastest pattern)
    for _, window in src.block_windows(1):
        data = src.read(1, window=window)
        # if you have nodata, you can mask it here; otherwise keep as-is
        bmin = int(data.min())
        bmax = int(data.max())
        if bmin < gmin: gmin = bmin
        if bmax > gmax: gmax = bmax
        count += 1
        if count % 5000 == 0:
            elapsed = time.time() - t0
            print(f"blocks={count:,}  min={gmin}  max={gmax}  elapsed={timedelta(seconds=int(elapsed))}")

elapsed = time.time() - t0
print("\n=== GLOBAL CHECK ===")
print("dtype:", dtype)
print("global min:", gmin)
print("global max:", gmax)
print("elapsed:", timedelta(seconds=int(elapsed)))
