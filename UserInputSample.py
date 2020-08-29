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
from numpy import load
import matplotlib


#Satellite image access

#date = datetime.now
# if type(inputfile)==tiff
#   geotiff.open
#   latlon = rasterio.warp.transform(tif.bounds)
#   
# if type(inputfile) = shape
#   ....
# 
#  
#else
#footprint = 

# footprint = 'POINT(-102.16 34.33)'
#footprint = 'footprint: "intersects(POINT(37.2480 -120.9997))"'
#returns product matches by product ID
#products = api.query(footprint,
#                    date=('20190601', date(2019, 6, 20)),
#                    area_relation='Intersects',
#                    platformname = 'Sentinel-2',
#                    cloudcoverpercentage=(0, 10)
#
#)


#apigeodf = api.to_geodataframe(products)
#newid = apigeodf['uuid'][0] #code for accesing id

#api = SentinelAPI('croblitos', 'LucklessMonkey$30', 'https://scihub.copernicus.eu/dhus')
#userpath = "userimages/ exampleuserpath"
#api.download('1d6bf68e-0f1f-4662-8b83-9d084a62ff57', directory_path=userpath)
# MyImagePath = userpath

"EXAMPLE CODE B/C NO USER INPUT"
examplecoord = (-120.3, 37.81)
exlon = examplecoord[1]
exlat = examplecoord[0]
exinputCRS = 'WGS84'
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

pointbuffer = 1000 #this is in 10 m units
boxbuffer = 20 # this is in 10 m units

#bbox = rasterio.warp.transform_bounds()

exx, exy = rasterio.warp.transform(exinputCRS, band4.crs, [exlat], [exlon]) # check if lat, lon should be flipped

row, col = band4.index(exx, exy)
c = col[0]
r = row[0]
print(r)
print(c)
top = c+pointbuffer
bot = c-pointbuffer
left = r-pointbuffer
right = r+pointbuffer

print(bot-top)
print(right-left)

ogRedBand = band4.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
ogNIRBand = band8.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
rededge = band5.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
ogGreenBand = band3.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
ogBlueBand = band2.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
wnnir = band8A.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
wswir1 = band11.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
wswir2 = band12.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')


#Read all bands into np array as float64
#RedBand = band4.read(1).astype('float64') #needs to be float64 for NDVI
#NIRBand = band8.read(1).astype('float64') #needs to be float64 for NDVI
#RedEdgeBand = band5.read(1).astype('float64') # do all these need to be float 64's??
#GreenBand = band3.read(1).astype('float64') 
#BlueBand = band2.read(1).astype('float64') 
#NNIRBand = band8A.read(1).astype('float64') 
#SWIR1Band = band11.read(1).astype('float64') 
#SWIR2Band = band12.read(1).astype('float64') 


#Stretch all 20m bands to 10980x10980
RedEdgeStretch = np.repeat(np.repeat(rededge,2, axis=0), 2, axis=1) 
SWIR1Stretch = np.repeat(np.repeat(wswir1,2, axis=0), 2, axis=1) 
SWIR2Stretch = np.repeat(np.repeat(wswir2,2, axis=0), 2, axis=1) 
NNIRStretch = np.repeat(np.repeat(wnnir,2, axis=0), 2, axis=1) 


#ogRedBand = RedBand
ogRedEdgeStretch = RedEdgeStretch
ogSWIR1Stretch = SWIR1Stretch
ogSWIR2Stretch = SWIR2Stretch
ogNNIRStretch= NNIRStretch
#ogNIRBand = NIRBand
#ogBlueBand = BlueBand
#ogGreenBand = GreenBand



# Calculate NDVI
nNDVI = np.where(
    (ogNIRBand+ogRedBand)==0., 
    0,
    (ogNIRBand-ogRedBand)/(ogNIRBand+ogRedBand)
)

# NDRE (NIR-RE)/(NIR+RE)
nNDRE = np.where(
    (ogNIRBand+ogRedEdgeStretch)==0.,
    0.,
    (ogNIRBand-ogRedEdgeStretch)/(ogNIRBand+ogRedEdgeStretch)
) 

# RECI (NIR/Red Edge)
nRECI = np.where(
    ogRedEdgeStretch==0.,
    0.,
    ogNIRBand/ogRedEdgeStretch
)

# SAVI ((NIR – Red) / (NIR + Red + L)) x (1 + L)
L = 0 #L is variable from -1 to 1. For high green vegetation, L is set to 0, whereas for low green vegetation, it is set to 1.
nSAVI = np.where(
    ((ogNIRBand+ogRedBand+L) * (1+L)) ==0.,
    0.,
    (ogNIRBand - ogRedBand) / ((ogNIRBand+ogRedBand+L) * (1+L))
)

# SIPI (NIR-Blue)/(NIR-Red)
nSIPI = np.where(
    (ogNIRBand-ogRedBand)==0.,
    0.,
    (ogNIRBand-ogBlueBand)/(ogNIRBand-ogRedBand)
)

# ARVI (NIR-(2*Red)+Blue)/(NIR + (2*Red) + Blue))
nARVI = np.where(
    (ogNIRBand+(2*ogRedBand)+ogBlueBand)==0.,
    0.,
    (ogNIRBand-(2*ogRedBand)+ogBlueBand)/(ogNIRBand+(2*ogRedBand)+ogBlueBand)
)

# GVMI (NIR+0.1) - (SWIR+0.2) // (NIR+0.1) + (SWIR+0.2) #currently attempt with SWIR1
nGVMI = np.where(
    ((ogNIRBand+0.1)+(ogSWIR1Stretch+0.2))==0.,
    0.,
    (ogNIRBand+0.1)-(ogSWIR1Stretch+0.2)/(ogNIRBand+0.1)+(ogSWIR1Stretch+0.2)
)

# NDMI Narrow NIR - (SWIR1 - SWIR2) / NarrowNIR + (SWIR1-SWIR2)
nNDMI = np.where(
    (ogNNIRStretch + (ogSWIR1Stretch-ogSWIR2Stretch))==0.,
    0.,
    (ogNNIRStretch - (ogSWIR1Stretch-ogSWIR2Stretch)) / (ogNNIRStretch + (ogSWIR1Stretch-ogSWIR2Stretch))
)
#NDMI limit to values between -1 and 1 for visibility
nNDMI = np.where(
    nNDMI>1.,
    0.,
    nNDMI
)
nNDMI = np.where(
    nNDMI<-1.,
    0.,
    nNDMI
)

# GCI (NIR/Green)
nGCI = np.where(
    ogGreenBand==0.,
    0.,
    ogNIRBand/ogGreenBand
)
# NDWI (B8A-B11)/(B11+B8A) THIS IS THE CORRECT ONE!!!!!! FOR MOISTURE MAPPING!
nNDWI = np.where(
    (ogSWIR1Stretch+ogNNIRStretch)==0.,
    0.,
    (ogNNIRStretch- ogSWIR1Stretch)/(ogSWIR1Stretch+ogNNIRStretch)
)

# MOISTURE INDEX (B8A-B11)/(B8A+B11)
nMI = np.where(
    (ogNNIRStretch+ogSWIR2Stretch)==0.,
    0.,
    (ogNNIRStretch-ogSWIR2Stretch)/(ogNNIRStretch+ogSWIR2Stretch)
)
#newTrueColor = np.stack((band2.read(1), band3.read(1), band4.read(1)), axis=0)
FalseColor = np.stack((ogSWIR2Stretch, ogSWIR1Stretch, ogBlueBand), axis=0)
Agriculture = np.stack((ogSWIR1Stretch, ogNIRBand, ogBlueBand), axis=0)
ColorInfrared = np.stack((ogNIRBand, ogRedBand, ogGreenBand), axis=0)
band2.close()
band3.close()
band4.close()
band5.close()
band8.close()
band8A.close()
band11.close()
band12.close()
# show stacked files of interest
#           ep.plot_rgb(newTrueColor, rgb=(0,1,2), figsize=(12,12), title='True Color')
#savetxt('NDVITEST.csv', NDVI, delimiter=',')
#savez_compressed('SampleImages/ndvi1.npz', NDVI)
#savez_compressed('SampleImages/ndre1.npz', NDRE)
#savez_compressed('SampleImages/savi1.npz', SAVI)
#savez_compressed('SampleImages/sipi1.npz', SIPI)
#savez_compressed('SampleImages/arvi1.npz', ARVI)
#savez_compressed('SampleImages/gvmi1.npz', GVMI)
#savez_compressed('SampleImages/ndmi1.npz', NDMI)
#savez_compressed('SampleImages/gci1.npz', GCI)
#savez_compressed('SampleImages/ndwi1.npz', NDWI)
#savez_compressed('SampleImages/mi1.npz', MI)
#   savez_compressed('SampleImages/ndvi3.npz', nNDVI)
#   savez_compressed('SampleImages/ndre3.npz', nNDRE)
#   savez_compressed('SampleImages/savi3.npz', nSAVI)
#savez_compressed('SampleImages/sipi2.npz', nSIPI)
#savez_compressed('SampleImages/arvi2.npz', nARVI)
#savez_compressed('SampleImages/gvmi2.npz', nGVMI)
#savez_compressed('SampleImages/ndmi2.npz', nNDMI)
#savez_compressed('SampleImages/gci2.npz', nGCI)
#   savez_compressed('SampleImages/ndwi3.npz', nNDWI)
#savez_compressed('SampleImages/mi2.npz', nMI)

#savez_compressed('SampleImages/truecolor.npz', newTrueColor)
#savez_compressed('SampleImages/falsecolor.npz', FalseColor)
#savez_compressed('SampleImages/agriculture.npz', Agriculture)
#savez_compressed('SampleImages/colorinfrared.npz', ColorInfrared)

tci = bandTCI.read(1)

matplotlib.image.imsave('sampletcitest.png', tci)