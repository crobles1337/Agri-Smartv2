import logging
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from sentinelloader import Sentinel2Loader
from shapely.geometry import Polygon







        


    
            
# GET THE FUNCTION TO EXTRACT COORDINATES, SEE IF IT WORKS FOR A SHAPE FILE
# GET THE FUNCTION TO CONVERT ANY COORDIANTES TO EPSG:4362







username = 'croblitos' 
pword = 'LucklessMonkey$30'

userdate = '2012-06-15'

sl = Sentinel2Loader('/notebooks/data/output/sentinelcache', 
                    username, pword,
                    apiUrl='https://scihub.copernicus.eu/apihub/', showProgressbars=True, loglevel=logging.DEBUG)

area = Polygon([(-120.55542, 37.9431), (-120.01211, 37.9431),
        (-120.01211, 37.4391), (-120.55542, 37.4391)])

geoTiffs = sl.getRegionHistory(area, 'B04', '10m', '2019-01-06', '2019-01-30', daysStep=5)
for geoTiff in geoTiffs:
    print('Desired image was prepared at')
    print(geoTiff)
    os.remove(geoTiff)

sl.getProductBandTiles(area, 'B04', '10m', userdate)
#https://github.com/flaviostutz/sentinelloader