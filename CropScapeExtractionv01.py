from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import geopandas
import shapely
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show_hist, show
import rasterio.features
import rasterio.warp
from affine import Affine
import time

"Step 0: Learn how to open the file"
"Step 1: Learn how to snip image. I should only be working be with pieces of this massive file for speed"
"Step 2: Learn how to access bands"
"Step 3: Learn how combine bands"
"Step 4: Learn how to georeference and move between coordinate systems. AKA, how to access wheat coordinates w/in image"


#Code to extract coordinates of crop from raster
seconds = time.time()


CropTifPath = "MercedWinterWheat.tif"

#src is a dataset object
src = rasterio.open(CropTifPath) #contains mask, nodata values, shape

#read np array of raw pixels
myCrop = src.read(1)

row = [0]*src.width
col = [0]*src.height
CropCoords = set()
CropExactx = []
CropExacty = []
srcboundsleft = src.bounds.left
srcboundstop = src.bounds.top
CropScapeRes = 30 #CropScape raster resolution in meters
seconds1 = time.time()
print("check1", seconds1-seconds)
"Be conscious of the offset when multiplying by 30, may end up not centered in pixel"

# Place crop coordinates into CropCoordList
nrow = myCrop.shape[0]
ncol = myCrop.shape[1]
CropCoordListX = [0]*(nrow*ncol)#try array
CropCoordListY = [0]*(nrow*ncol)
count = 0
for i in range(nrow):
    for j in range(ncol):
        if(myCrop[i, j] != 0):
            xCo, yCo = src.xy(i, j)
            CropCoordListX[count] = (xCo)
            CropCoordListY[count] = (yCo)
            count += count
# Place coordinates in crop set
##for i in range(len(row)):
##    for j in range(len(col)):
##        x,y = ((srcboundsleft +(CropScapeRes*i)), (srcboundstop - (CropScapeRes*j)))
##        row[i], col[j] = src.index(x, y)    
##        #extract non-zero values, corresponds to crop filled spaces
##        if (row[i] !=0 and col[j]!=0):
##            if(myCrop[row[i], col[j]] !=0):
##                cox, coy = src.xy(row[i], col[j])
##                CropExactx.append(cox) #List of x coordinates
##                CropExacty.append(coy) #List of y coordinates
##                #CropCoords.add((row[i],col[j])) #set of the pixel Array Indices corresponding to points in georeferenced space

satelliteCRS = 'EPSG:32611'

seconds2 = time.time()
print("check2", seconds2-seconds)
# Convert coordinates to EPSG:32611
xyCropCoords = rasterio.warp.transform(
    src_crs=src.crs, dst_crs=satelliteCRS, xs = CropCoordListX, ys = CropCoordListY)

CropSet = set()
for i in range(len(xyCropCoords[0])):
    CropSet.add((xyCropCoords[1][i], xyCropCoords[0][i])) 



"OLD CODE"
# Place coordinates in crop set
##for i in range(len(row)):
##    for j in range(len(col)):
##        x,y = ((srcboundsleft +(CropScapeRes*i)), (srcboundstop - (CropScapeRes*j)))
##        row[i], col[j] = src.index(x, y)    
##        #extract non-zero values, corresponds to crop filled spaces
##        if (row[i] !=0 and col[j]!=0):
##            if(myCrop[row[i], col[j]] !=0):
##                cox, coy = src.xy(row[i], col[j])
##                CropExactx.append(cox) #List of x coordinates
##                CropExacty.append(coy) #List of y coordinates
##                #CropCoords.add((row[i],col[j])) #set of the pixel Array Indices corresponding to points in georeferenced space
