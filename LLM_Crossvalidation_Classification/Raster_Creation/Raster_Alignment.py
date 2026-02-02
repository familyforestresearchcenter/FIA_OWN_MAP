import math
import gc
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

OWNERSHIP = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\US_Ownership_Landcover_9_7_2025.tif"
NLCD_SRC  = r"F:\OwnershipMap\New_Script\Error_Calculations\Annual_NLCD_LndCnf_2020_CU_C1V0.tif"
NLCD_OUT  = r"F:\OwnershipMap\New_Script\Error_Calculations\Annual_NLCD_LndCnf_2020_ALIGNED_TO_OWNERSHIP.tif"

SUPER_TILE = 4096  # same as your other scripts

with rio.open(OWNERSHIP) as ref, rio.open(NLCD_SRC) as src:

    profile = ref.profile.copy()
    profile.update(
        dtype="uint8",
        count=1,
        compress="zstd",
        tiled=True,
        BIGTIFF="YES",
        blockxsize=1024,
        blockysize=1024,
    )

    height, width = ref.height, ref.width
    transform = ref.transform
    crs = ref.crs

    tiles_x = math.ceil(width / SUPER_TILE)
    tiles_y = math.ceil(height / SUPER_TILE)
    total_tiles = tiles_x * tiles_y

    print(f"Reprojecting NLCD confidence to ownership grid")
    print(f"Tiles: {tiles_x} x {tiles_y} = {total_tiles}")

    with rio.open(NLCD_OUT, "w", **profile) as dst, \
         tqdm(total=total_tiles, unit="tile", desc="Aligning NLCD") as pbar:

        for ty in range(tiles_y):
            y0 = ty * SUPER_TILE
            y1 = min((ty + 1) * SUPER_TILE, height)

            for tx in range(tiles_x):
                x0 = tx * SUPER_TILE
                x1 = min((tx + 1) * SUPER_TILE, width)

                window = Window(x0, y0, x1 - x0, y1 - y0)

                # Destination buffer for this tile
                dst_arr = np.zeros((int(window.height), int(window.width)), dtype=np.uint8)

                # Compute destination transform for this window
                dst_transform = rio.windows.transform(window, transform)

                reproject(
                    source=rio.band(src, 1),
                    destination=dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                )

                dst.write(dst_arr, 1, window=window)

                del dst_arr
                gc.collect()
                pbar.update(1)

print("✅ NLCD confidence successfully aligned (tiled)")
