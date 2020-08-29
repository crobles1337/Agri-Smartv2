import logging
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from SentinelLoadstutz import Sentinel2Loader
from shapely.geometry import Polygon
from SentinelLoadstutzv2 import Sentinel2Loaderv2





    
            
# GET THE FUNCTION TO EXTRACT COORDINATES, SEE IF IT WORKS FOR A SHAPE FILE
# GET THE FUNCTION TO CONVERT ANY COORDIANTES TO EPSG:4362







username = 'croblitos' 
pword = 'LucklessMonkey$30'

userdate = '2020-08-05'

sl = Sentinel2Loaderv2('', 
                    username, pword,
                    apiUrl='https://scihub.copernicus.eu/apihub/', showProgressbars=True, loglevel=logging.DEBUG, cacheApiCalls=True, savepath='rtest/')

area = Polygon([(-120.55542, 37.9431), (-120.01211, 37.9431),
        (-120.01211, 37.4391), (-120.55542, 37.4391)])

#geoTiffs = sl.getRegionHistory(area, 'B04', '10m', '2019-01-06', '2019-01-30', daysStep=1)
#for geoTiff in geoTiffs:
#    print('Desired image was prepared at')  
#    print(geoTiff)
#    os.remove(geoTiff)

fpath = sl.getProductBandTiles(area, 'B08', '10m', userdate)
print("this is the list", fpath)

f2path = sl.getProductBandTiles(area, 'B04', '10m', userdate)
print("this is the next list", f2path)



blist = [('B04', '10m'), ('B08', '10m'), ('B03', '10m'), ('B02', '10m'),('B05', '20m'),  ('B8A', '20m'), ('B11', '20m'), ('B12', '20m'), ('TCI', '10m') ]
bdict = {'B04': '10m', 'B08': '10m', 'B03': '10m', 'B02': '10m', 'B05': '20m',  'B8A': '20m', 'B11': '20m', 'B12': '20m', 'TCI': '10m' }


#https://github.com/flaviostutz/sentinelloader






#url = 
#downFile(, , username, pword)










def downFile(url, filepath, user, password):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, "wb") as f:
        logger.debug("Downloading %s to %s" % (url, filepath))
        response = requests.get(url, auth=(user, password), stream=True)
        if response.status_code != 200:
            raise Exception("Could not download file. status=%s" % response.status_code)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
                sys.stdout.flush()