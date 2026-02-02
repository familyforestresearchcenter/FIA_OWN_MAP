import os
import gc
import math
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import rasterio as rio
from rasterio.windows import Window

# ------------------------------------------------------------
# Environment / scratch
# ------------------------------------------------------------
TMP_DIR = os.environ.get("TMP_DIR", "/tmp")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
OWNERSHIP_RASTER = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\US_Ownership_Landcover_9_7_2025.tif"
NLCD_CONF_RASTER = r"F:\OwnershipMap\New_Script\Error_Calculations\Annual_NLCD_LndCnf_2020_ALIGNED_TO_OWNERSHIP.tif"
LOOKUP_CSV = r"F:\OwnershipMap\New_Script\Test_ENV\SCRIPTS\Vance_Harris_May_2022\Error_Calculation\llm_classifier\FIA_Ownership_Map_Data_Dictionary_NEW_COLUMNS - FIA_Ownership_Map_Data_Dictionary_NEW_COLUMNS.csv"

OUTPUT_RASTER = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\US_Ownership_Landcover_confidence_v1_q100.tif"

SUPER_TILE = 4096  # pixels

# ------------------------------------------------------------
# Build ownership confidence LUT (Map_Value_Code -> confidence 0–100)
# ------------------------------------------------------------
def build_confidence_lut(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    # DMP-adjusted ownership accuracies (0–1)
    owncd_accuracy = {
        -1: 1.00,   # Known unknown
        25: 0.81, 
        31: 0.86,
        32: 0.84,
        41: 0.86,
        42: 0.86,
        43: 0.86,
        44: 0.95,
        45: 0.94,
    }

    # Convert to explicit 0–100 scale
    owncd_conf_100 = {
        k: int(round(v * 100))
        for k, v in owncd_accuracy.items()
    }

    lut = np.zeros(256, dtype=np.uint8)

    for _, row in df.iterrows():
        map_val = int(row["Map_Value_Code"])
        owner = int(row["owner_clean"])
        lut[map_val] = owncd_conf_100.get(owner, 0)

    # Hard validation
    if lut.dtype != np.uint8 or lut.max() > 100:
        raise RuntimeError("Invalid ownership confidence LUT (must be uint8, 0–100)")

    return lut


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    # Safety: remove existing output
    if os.path.exists(OUTPUT_RASTER):
        print("⚠️ Existing output raster found — deleting it")
        os.remove(OUTPUT_RASTER)

    print("🔧 Building ownership confidence LUT (0–100)...")
    confidence_lut = build_confidence_lut(LOOKUP_CSV)

    print("📂 Opening rasters...")
    with rio.open(OWNERSHIP_RASTER) as own_src, \
         rio.open(NLCD_CONF_RASTER) as nlcd_src:

        # ----------------------------------------------------
        # Alignment checks (mandatory)
        # ----------------------------------------------------
        if own_src.transform != nlcd_src.transform:
            raise ValueError("Raster transforms do not match")
        if own_src.crs != nlcd_src.crs:
            raise ValueError("Raster CRS do not match")
        if own_src.width != nlcd_src.width or own_src.height != nlcd_src.height:
            raise ValueError("Raster dimensions do not match")

        height, width = own_src.height, own_src.width
        transform, crs = own_src.transform, own_src.crs

        tiles_x = math.ceil(width / SUPER_TILE)
        tiles_y = math.ceil(height / SUPER_TILE)
        total_tiles = tiles_x * tiles_y

        print(f"🧮 Raster size: {width} x {height}")
        print(f"🧩 Processing {total_tiles} supertiles ({tiles_x} x {tiles_y})")

        # ----------------------------------------------------
        # OUTPUT PROFILE (COG-safe, uint8)
        # ----------------------------------------------------
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,

            # COG requirements
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,

            # Compression
            "compress": "LZW",
            "predictor": 2,

            # Large raster support
            "BIGTIFF": "YES",

            # Performance
            "NUM_THREADS": "ALL_CPUS",
        }

        env_kwargs = dict(
            GDAL_CACHEMAX=1024,
            VSI_CACHE="TRUE",
            VSI_CACHE_SIZE=100_000_000,
        )

        start_time = time.time()

        print("🚀 Building final confidence raster (0–100)...")
        with rio.Env(**env_kwargs):
            with rio.open(OUTPUT_RASTER, "w", **profile) as dst, \
                 tqdm(
                     total=total_tiles,
                     unit="supertile",
                     desc="Building confidence raster",
                     dynamic_ncols=True,
                     smoothing=0.05,
                 ) as pbar:

                for ty in range(tiles_y):
                    y0 = ty * SUPER_TILE
                    y1 = min((ty + 1) * SUPER_TILE, height)

                    for tx in range(tiles_x):
                        x0 = tx * SUPER_TILE
                        x1 = min((tx + 1) * SUPER_TILE, width)

                        window = Window(x0, y0, x1 - x0, y1 - y0)

                        # Read blocks
                        own_block = own_src.read(1, window=window).astype(np.uint8)
                        nlcd_block = nlcd_src.read(1, window=window).astype(np.uint8)

                        # Ownership confidence (0–100)
                        own_100 = confidence_lut[own_block].astype(np.uint16)

                        # NLCD confidence: cap semantically at 100
                        nlcd_100 = np.minimum(nlcd_block, 100).astype(np.uint16)

                        # Combine confidences (0–100)
                        final_100 = (own_100 * nlcd_100) // 100
                        final_100 = final_100.astype(np.uint8)

                        # Hard guard
                        if final_100.max() > 100:
                            raise RuntimeError("Final confidence exceeds 100")

                        dst.write(final_100, 1, window=window)

                        # Progress
                        pbar.update(1)

                        del own_block, nlcd_block, own_100, nlcd_100, final_100
                        gc.collect()

    runtime = time.time() - start_time

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------
    with rio.open(OUTPUT_RASTER) as src:
        print("✅ Output dtype:", src.dtypes)
        print("📊 Output range: 0–100")
        print("📦 Output size (GB):", os.path.getsize(OUTPUT_RASTER) / 1e9)

    print(f"\n⏱️ Total runtime: {str(timedelta(seconds=int(runtime)))}")
    print(f"✅ Final confidence raster written to:\n   {OUTPUT_RASTER}")
