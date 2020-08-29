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
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from rasterio.plot import plotting_extent
from numpy import save
from numpy import savetxt
from numpy import savez_compressed

"Step 0: Learn how to save, both to file, and to jpeg"
"Step 1: Learn how to make plots more visually telling"
"Step 2: Learn how to make band/ML algorithm ignore any 0 values"
"Step 3: Learn how to make valid input for ML, can i just save pixel values as list?"
"Step 4: Translating pixel count to acre/area/sq. meter count in order to attach yield value, per acre."
"Possibly use nodata values which are contained a rasterio tiff to label 0 values"
"Deal with how indices that include valid negative numbers, can deal with 0'd out values, possibly usinng nodata"

#Satellite image access
imagePath01 = "SentImages_Training\SatExtract1_Test\S2B_MSIL1C_20180904T184019_N0206_R070_T11SKB_20180904T233822\S2B_MSIL1C_20180904T184019_N0206_R070_T11SKB_20180904T233822.SAFE\GRANULE\L1C_T11SKB_A007818_20180904T185252\IMG_DATA\T11SKB_20180904T184019_"
imagePath02 = "SentImages_Training\SatExtract2\S2A_MSIL1C_20190619T184921_N0207_R113_T10SGG_20190620T000501\S2A_MSIL1C_20190619T184921_N0207_R113_T10SGG_20190620T000501.SAFE\GRANULE\L1C_T10SGG_A020845_20190619T185641\IMG_DATA\T10SGG_20190619T184921_"
imagePath03 = 'SentImages_Training\Imagery\Sat2_Merced_2019_06_19\SatExtract2\S2A_MSIL1C_20190619T184921_N0207_R113_T10SGG_20190620T000501\S2A_MSIL1C_20190619T184921_N0207_R113_T10SGG_20190620T000501.SAFE\GRANULE\L1C_T10SGG_A020845_20190619T185641\IMG_DATA\T10SGG_20190619T184921_'
#Extract all bands of interest
MyImagePath = imagePath03

band2 = rasterio.open(MyImagePath + "B02.jp2", driver = 'JP2OpenJPEG') #blue - 10m
band3 = rasterio.open(MyImagePath + "B03.jp2",  driver = 'JP2OpenJPEG') #green - 10m
band4 = rasterio.open(MyImagePath + "B04.jp2",  driver = 'JP2OpenJPEG') #red - 10m
band5 = rasterio.open(MyImagePath + "B05.jp2",  driver = 'JP2OpenJPEG') #red edge close to center - 20m, 5990
band8 = rasterio.open(MyImagePath + "B08.jp2",  driver = 'JP2OpenJPEG') #nir - 10m, 10980
band8A = rasterio.open(MyImagePath + "B8A.jp2",  driver = 'JP2OpenJPEG') #narrownir - 20m, 5990
#band10 = rasterio.open(MyImagePath + "B10.jp2",  driver = 'JP2OpenJPEG') #SWIR-cirrus - 60m
band11 = rasterio.open(MyImagePath + "B11.jp2",  driver = 'JP2OpenJPEG') #swir1 - 20m
band12 = rasterio.open(MyImagePath + "B12.jp2",  driver = 'JP2OpenJPEG') #swir2 - 20m
bandTCI = rasterio.open(MyImagePath + "TCI.jp2",  driver = 'JP2OpenJPEG') #swir2 - 10m

# Begin extract coordinates of crop from CropScape geotiff
CropTifPath = "CropScapeRaster\MercedWinterWheat.tif"

#src is a dataset object
src = rasterio.open(CropTifPath) #contains mask, nodata values, shape

#read np array of raw pixels
myCrop = src.read(1)

# Place crop coordinates into CropCoordList
nrow = myCrop.shape[0]
ncol = myCrop.shape[1]
CropCoordListX = [0]*(nrow*ncol)
CropCoordListY = [0]*(nrow*ncol)
count = 0
for i in range(nrow):
    for j in range(ncol):
        if(myCrop[i, j] != 0):
            xCo, yCo = src.xy(i,j)
            CropCoordListX[count] = xCo
            CropCoordListY[count] = yCo
            count = count+1

# Convert CropScape coordinates to Sentinel image crs
satelliteCRS = band2.crs
xyCropCoords = rasterio.warp.transform(
    src_crs=src.crs, dst_crs=satelliteCRS, xs = CropCoordListX, ys = CropCoordListY)
# Place coordinates in CropSet
CropSet = set()
for i in range(len(xyCropCoords[0])):
    CropSet.add((xyCropCoords[1][i], xyCropCoords[0][i])) 

#Read all bands into np array as float64
RedBand = band4.read(1).astype('float64') #needs to be float64 for NDVI
NIRBand = band8.read(1).astype('float64') #needs to be float64 for NDVI
RedEdgeBand = band5.read(1).astype('float64') # do all these need to be float 64's??
GreenBand = band3.read(1).astype('float64') 
BlueBand = band2.read(1).astype('float64') 
NNIRBand = band8A.read(1).astype('float64') 
SWIR1Band = band11.read(1).astype('float64') 
SWIR2Band = band12.read(1).astype('float64') 

# Create pixel indices set corresponding to valid crop coordinates using B4 as an arbitrary, 10m band
pixelSet = set()
for i in (CropSet):
    lon = i[0]
    lat = i[1]
    npy, npx = band4.index(lat, lon)
    pixelSet.add((npy, npx))

# Create new set w/ wider resolution 3x3 per pixel, b/c coordinates are in 30x30 while image in 10x10
Pixel30x30Count = 0
checkSet = set() 
width = band4.width
height = band4.height
for i in pixelSet:
    if (i[0]>0):
        if (i[0]<width):
            if(i[1]>0):
                if(i[1]<height):
                    Pixel30x30Count = Pixel30x30Count+1
                    checkSet.add((i[0], i[1]))
                    checkSet.add((i[0]+1, i[1]))
                    checkSet.add((i[0]-1, i[1]))
                    checkSet.add((i[0], i[1]+1))
                    checkSet.add((i[0]+1, i[1]+1))
                    checkSet.add((i[0]-1, i[1]+1))
                    checkSet.add((i[0], i[1]-1))
                    checkSet.add((i[0]+1, i[1]-1))
                    checkSet.add((i[0]-1, i[1]-1))   


#Stretch all 20m bands to 10980x10980
RedEdgeStretch = np.repeat(np.repeat(RedEdgeBand,2, axis=0), 2, axis=1) 
SWIR1Stretch = np.repeat(np.repeat(SWIR1Band,2, axis=0), 2, axis=1) 
SWIR2Stretch = np.repeat(np.repeat(SWIR2Band,2, axis=0), 2, axis=1) 
NNIRStretch = np.repeat(np.repeat(NNIRBand,2, axis=0), 2, axis=1) 

print(Pixel30x30Count, "30X30METERS: ")

# Blackout band pixels not in cropset
for i in range(width):
    for j in range(height):
        if ((i,j) not in checkSet):
            RedBand[i,j] = 0.
            RedEdgeStretch[i,j] = 0.
            SWIR1Stretch[i,j] = 0.
            SWIR2Stretch[i,j] = 0.
            NNIRStretch[i,j] = 0.
            NIRBand[i,j] = 0.
            BlueBand[i,j] = 0.
            GreenBand[i,j] = 0.

# Calculate NDVI
NDVI = np.where(
    (NIRBand+RedBand)==0., 
    0,
    (NIRBand-RedBand)/(NIRBand+RedBand)
)

# NDRE (NIR-RE)/(NIR+RE)
NDRE = np.where(
    (NIRBand+RedEdgeStretch)==0.,
    0.,
    (NIRBand-RedEdgeStretch)/(NIRBand+RedEdgeStretch)
) 

# RECI (NIR/Red Edge)
RECI = np.where(
    RedEdgeStretch==0.,
    0.,
    NIRBand/RedEdgeStretch
)

# SAVI ((NIR – Red) / (NIR + Red + L)) x (1 + L)
L = 0 #L is variable from -1 to 1. For high green vegetation, L is set to 0, whereas for low green vegetation, it is set to 1.
SAVI = np.where(
    ((NIRBand+RedBand+L) * (1+L)) ==0.,
    0.,
    (NIRBand - RedBand) / ((NIRBand+RedBand+L) * (1+L))
)

# SIPI (NIR-Blue)/(NIR-Red)
SIPI = np.where(
    (NIRBand-RedBand)==0.,
    0.,
    (NIRBand-BlueBand)/(NIRBand-RedBand)
)

# ARVI (NIR-(2*Red)+Blue)/(NIR + (2*Red) + Blue))
ARVI = np.where(
    (NIRBand+(2*RedBand)+BlueBand)==0.,
    0.,
    (NIRBand-(2*RedBand)+BlueBand)/(NIRBand+(2*RedBand)+BlueBand)
)

# GVMI (NIR+0.1) - (SWIR+0.2) // (NIR+0.1) + (SWIR+0.2) #currently attempt with SWIR1
GVMI = np.where(
    ((NIRBand+0.1)+(SWIR1Stretch+0.2))==0.,
    0.,
    (NIRBand+0.1)-(SWIR1Stretch+0.2)/(NIRBand+0.1)+(SWIR1Stretch+0.2)
)

# NDMI Narrow NIR - (SWIR1 - SWIR2) / NarrowNIR + (SWIR1-SWIR2)
NDMI = np.where(
    (NNIRStretch + (SWIR1Stretch-SWIR2Stretch))==0.,
    0.,
    (NNIRStretch - (SWIR1Stretch-SWIR2Stretch)) / (NNIRStretch + (SWIR1Stretch-SWIR2Stretch))
)
#NDMI limit to values between -1 and 1 for visibility
NDMI = np.where(
    NDMI>1.,
    0.,
    NDMI
)
NDMI = np.where(
    NDMI<-1.,
    0.,
    NDMI
)

# GCI (NIR/Green)
GCI = np.where(
    GreenBand==0.,
    0.,
    NIRBand/GreenBand
)
# NDWI (B3-B8)/(B3+B8)
NDWI = np.where(
    (GreenBand+NNIRStretch)==0.,
    0.,
    (GreenBand-NNIRStretch)/(GreenBand+NNIRStretch)
)

# MOISTURE INDEX (B8A-B11)/(B8A+B11)
MI = np.where(
    (NNIRStretch+SWIR2Stretch)==0.,
    0.,
    (NNIRStretch-SWIR2Stretch)/(NNIRStretch+SWIR2Stretch)
)

#newTrueColor = np.stack((band2.read(1), band3.read(1), band4.read(1)), axis=0)
#FalseColor = np.stack((SWIR2Stretch, SWIR1Stretch, BlueBand), axis=0)
#Agriculture = np.stack((SWIR1Stretch, NIRBand, BlueBand), axis=0)
#ColorInfrared = np.stack((NIRBand, RedBand, GreenBand), axis=0)




# Save images as TIFF file

#ndviImage = rasterio.open('SamplesImages/NDVISample_2019_06_19', 'w', driver = 'Gtiff',
#                            width = band4.width,
#                            height = band4.height,
#                            count = 1,
#                            crs=band4.crs,
#                            transform = band4.transform,
#                            dtype='float64') #experiment with crs parameter, may be easy way to tranform/access coordinates
#ndviImage.write(NDVI, 1)
#ndviImage.close()



# Open Saved tiff files
## ndvi = rasterio.open('ndvifilepath.tif')

# show files of interest
##plot.show(, cmap='RbYlGn')

# show stacked files of interest
##ep.plot_rgb(newTrueColor, rgb=(0,1,2), figsize=(12,12), title='True Color')
#savetxt('NDVITEST.csv', NDVI, delimiter=',')
savez_compressed('NDVICOMPRESSEDTEST.npz', NDVI)

#save('nparrays/NDVI_Test.npy', NDVI)
#save('nparrays/RECI_Test.npy', RECI)
#save('nparrays/NDRE_Test.npy', NDRE)
#save('nparrays/SAVI_Test.npy', SAVI)
#save('nparrays/SIPI_Test.npy', SIPI)
#save('nparrays/ARVI_Test.npy', ARVI)
#save('nparrays/GVMI_Test.npy', GVMI)
#save('nparrays/NDMI_Test.npy', NDMI)
#save('nparrays/NDWI_Test.npy', NDWI)
#save('nparrays/GCI_Test.npy', GCI)
#save('nparrays/MI_Test.npy', MI)

#  try with np stacks!
#save('nparrays/TrueColor_Test.npy', newTrueColor)
#save('nparrays/FalseColor_Test.npy', FalseColor)
#save('nparrays/Agriculture_Test.npy', Agriculture)
#save('nparrays/ColorInfrared_Test.npy', ColorInfrared)