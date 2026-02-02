import rasterio as rio

OWNERSHIP_RASTER = r"F:\OwnershipMap\New_Script\Test_ENV\OUTPUTS\FINAL\Round6\US_Ownership_Landcover_9_7_2025.tif"
NLCD_CONF_RASTER = r"F:\OwnershipMap\New_Script\Error_Calculations\Annual_NLCD_LndCnf_2020_ALIGNED_TO_OWNERSHIP.tif"


with rio.open(OWNERSHIP_RASTER) as a, rio.open(NLCD_CONF_RASTER) as b:
    print(a.crs)
    print(b.crs)
    print(a.transform)
    print(b.transform)
    print(a.width, a.height)
    print(b.width, b.height)
    print(a.bounds)
    print(b.bounds)