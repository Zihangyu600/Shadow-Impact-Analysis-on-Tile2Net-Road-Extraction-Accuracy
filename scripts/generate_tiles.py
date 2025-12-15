from tile2net import Raster
import geopandas as gpd


location_ny = '40.4978740560533,-74.2550385911449,40.9151395390951,-73.6996327792974'
location_ny_small = '40.714,-73.980,40.744,-74.010'

raster = Raster(
    location=location_ny_small,
    zoom = 19,
    name='new york',
    output_dir=r"D:\data\outputdir",
    dump_percent=100,

)
raster

raster.generate(8)

raster.inference()
