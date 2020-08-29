import os
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
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from rasterio.plot import plotting_extent
from numpy import save
from numpy import savetxt
from numpy import savez_compressed
from numpy import load
import csv
import matplotlib
import imageio
from Bundle2Helpers import *


# def main()
# for path in os.list('SentImages_Training\Imagery'):
#   inputname = path
#   ExtractSet(inputname)


def main(): # def ExtractSet():
    BundleGenerate(inputname = 'Input1_Yakima_2019')

def BundleGenerate(inputname, datatype='train', Crop='Wheat', xStats = True): 
    datapath = os.path.join(Crop, 'data') #static
    imagerydir = 'Imagery' #static
    rasterdir = 'Rasters' #static

    if '2019' in inputname:
        start = '2019-01-01' # will prolly be this
        end = '2019-08-15' # will prolly be to mid august
    if '2018' in inputname:
        start = '2018-01-01' # will prolly be this
        end = '2018-08-15' # will prolly be to mid august

    #Crop = 'Wheat' #dynamic
    # create main directorya
    MainDirectory = MakeMainDir(datapath, inputname, datatype=datatype)
    ImagesDir = os.path.join('SentImages_Training\Imagery', inputname, imagerydir)
    RasterDir = os.path.join('SentImages_Training\Imagery', inputname, rasterdir)
    
    imagelist = GetImages(ImagesDir)
    rasterlist = GetRasters(RasterDir, Crop)
    SegList = seglist(imagelist, 3)
    cplist = list()
    cpbools = list()
    for i in range(len(SegList)): # iterating through 1 dated image (3 bands at a time)      
        #first go, should create equal number of folders to # of counties which==len(rasterlist)
        if i==0:
            for raster in rasterlist:
                headtail = os.path.split(raster)  
                tail = GetTail(raster)
                ras = RemoveExt(tail)
                countypath = os.path.join(MainDirectory, ras)  #if the rastername is something like, year,crop,county
                os.mkdir(countypath)
                cp, lon, lat, cpbool = GetCropMask(imagelist, raster)
                cplist.append(cp)
                cpbools.append(cpbool)
                LonLat = 'lon={lon}&lat={lat}'.format(lon=str(lon), lat=str(lat))
                LatLon = (lat, lon)
                if xStats==True:
                    GetSoil(LonLat, countypath)
                    GetHistWeather(LatLon, start, end, countypath)
        ct = 0
        for raster in rasterlist: #iterates counties, should equal # of counties. Crop already specified           
            if cpbools[ct] == True:
                band4, B4Red, B5RE, B8NIR = openbands(SegList, i)
                datepath = MakeDatePath(raster, MainDirectory, SegList, i) #makes datepath and saves path to DatePath                       
                ###########################################################
                ZB4Red, ZB5RE, ZB8NIR = ApplyStatMask(B4Red, B5RE, B8NIR, cplist[ct], band4) # 
                ##############################################
                ct = ct + 1         
                ndvi, ndre, reci, ccci = GetIndicesnew(ZB4Red, ZB5RE, ZB8NIR)
                savedata(ndvi, ndre, reci, ccci, band4, datepath)
                band4.close()


def GetCropMask(Images, raster):
    with rasterio.open(Images[0], 'r') as band4:
        
        Cropsrc = rasterio.open(raster)
        CropRaster = Cropsrc.read(1) # np array
        xyCropCoords = GetCropCoordinates(CropRaster, Cropsrc, band4) # already converted
        Cropxy = xytoSet(xyCropCoords)
        CropPixelsPre = GetCropPixelsPre(Cropxy, band4)
        if len(CropPixelsPre)>100:
            CropPixels, pixel30x30count = GetCropPixels(CropPixelsPre, band4)
        else:
            CropPixels = [0]
            Lon = 0
            Lat = 0
            return CropPixels, Lon, Lat, False
        #get center of COUNTY using EPSG coordinates
        bounds = Cropsrc.bounds
        nbounds = rasterio.warp.transform_bounds(Cropsrc.crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)
        left = nbounds[0] #lon
        bottom = nbounds[1] # lat
        right = nbounds[2] #lon
        top = nbounds[3] #lat

        Lon = left + (right-left)/2
        Lat =  bottom + (top-bottom)/2
        if len(CropPixels)>100:
            cpbool = True
        else:
            cpbool = False


    return CropPixels, Lon, Lat, cpbool




def GetImages(directory):
    imagelist = list() # list of filepaths
    for filename in os.listdir(directory):
        for filenew in os.listdir(os.path.join(directory, filename)):
            if filenew.endswith(".jp2"):
                newfile = os.path.join(directory, filename, filenew)
                imagelist.append(newfile)
    return imagelist
def GetRasters(RasterDir, Crop):
    rasterlist = list() # list of filepaths
    RasterPath = os.path.join(RasterDir, Crop)
    for filename in os.listdir(RasterPath):
        filenew = os.path.join(RasterPath, filename)
        rasterlist.append(filenew)
    return rasterlist
def seglist(imagelist, seg_length=3):
    seglist = list()
    [seglist.append(imagelist[x:x+seg_length]) for x in range(0,len(imagelist),seg_length)]
    return seglist
def openbands(seglist, index):
    Path4 = seglist[index][0]
    Path5 = seglist[index][1]
    Path8 = seglist[index][2]
    band4 = rasterio.open(Path4,  driver = 'JP2OpenJPEG') #red - 10m
    band5 = rasterio.open(Path5,  driver = 'JP2OpenJPEG') #red edge - 20m, 5990
    band8 = rasterio.open(Path8,  driver = 'JP2OpenJPEG') #nir - 10m, 10980
    B4Red = band4.read(1).astype('float64') 
    B8NIR = band8.read(1).astype('float64')
    B5RE = band5.read(1).astype('float64') 
    B5RE = np.repeat(np.repeat(B5RE,2, axis=0), 2, axis=1)
    band5.close()
    band8.close()
    return band4, B4Red, B5RE, B8NIR
    #leave band4 b/c i think it's used later
    #   either turn these to with opens, or make sure to close them at the end
# Convert CropScape coordinates to Sentinel image crs
def crsConvert(x, y, Cropsrc, band10m):
    satelliteCRS = band10m.crs
    xyCropCoords = rasterio.warp.transform(
        src_crs=Cropsrc.crs, dst_crs=satelliteCRS, xs = x, ys = y)
    return xyCropCoords
def GetCropCoordinates(CropRaster, Cropsrc, band10m):
    nrow = CropRaster.shape[0]
    ncol = CropRaster.shape[1]
    CropCoordListX = [0]*(nrow*ncol)
    CropCoordListY = [0]*(nrow*ncol)
    count = 0
    for i in range(nrow):
        for j in range(ncol):
            if(CropRaster[i, j] != 0):
                xCo, yCo = Cropsrc.xy(i,j)
                CropCoordListX[count] = xCo
                CropCoordListY[count] = yCo
                count = count+1
    xyCropCoords = crsConvert(x=CropCoordListX, y=CropCoordListY, Cropsrc=Cropsrc, band10m=band10m)
    return xyCropCoords
# Place coordinates in Cropxy, a set
def xytoSet(xyCropCoords):
    Cropxy = set()
    for i in range(len(xyCropCoords[0])):
        Cropxy.add((xyCropCoords[1][i], xyCropCoords[0][i])) 
    return Cropxy
# Create pixel indices set corresponding to valid crop coordinates using B4 as an arbitrary, 10m band
def GetCropPixelsPre(Cropxy, band10m): #band10m=band4
    CropPixelsPre = set()
    for i in (Cropxy):
        lon = i[0]
        lat = i[1]
        npy, npx = band10m.index(lat, lon)
        CropPixelsPre.add((npy, npx))
    return CropPixelsPre
# Create new set w/ wider resolution 3x3 per pixel, b/c coordinates are in 30x30 while image in 10x10
def GetCropPixels(CropPixelsPre, band10m):
    Pixel30x30Count = 0
    CropPixels = set() 
    width = band10m.width
    height = band10m.height
    for i in CropPixelsPre:
        if (i[0]>0):
            if (i[0]<width):
                if(i[1]>0):
                    if(i[1]<height):
                        Pixel30x30Count = Pixel30x30Count+1
                        CropPixels.add((i[0], i[1]))
                        CropPixels.add((i[0]+1, i[1]))
                        CropPixels.add((i[0]-1, i[1]))
                        CropPixels.add((i[0], i[1]+1))
                        CropPixels.add((i[0]+1, i[1]+1))
                        CropPixels.add((i[0]-1, i[1]+1))
                        CropPixels.add((i[0], i[1]-1))
                        CropPixels.add((i[0]+1, i[1]-1))
                        CropPixels.add((i[0]-1, i[1]-1))
    print(len(CropPixels), "lencropixels")
    print(Pixel30x30Count, "pix30x30")
    #HUGE, CROPIXELS IS A TUPLE IN ONE, AND SET IN THE OTHER!!!

    return CropPixels, Pixel30x30Count
# Blackout band pixels not in cropset
def ApplyMask(B4Reds, B5REs, B8NIRs, CropPixelss, band10ms):
    ncount = 0
    mcount = 0
    
    for i in range(band10ms.width):
        for j in range(band10ms.height):
            if ((i,j) not in CropPixelss):
                B4Reds[i,j] = 0.
                B5REs[i,j] = 0.
                B8NIRs[i,j] = 0.
    return B4Reds, B5REs, B8NIRs

#returns altered shape, only non-zero values
def ApplyStatMask(B4Reds, B5REs, B8NIRs, CropPixelss, band10ms):
    ncount = 0
    mcount = 0
    B4Mask = list()
    B5Mask = list()
    B8Mask = list()


    for i in range(band10ms.width):
        for j in range(band10ms.height):
            if ((i,j) in CropPixelss):
                #check for zero values/ any values off the map
                if B4Reds[i,j]!=0:
                    B4Mask.append(B4Reds[i,j])
                    B5Mask.append(B5REs[i,j])
                    B8Mask.append(B8NIRs[i,j])
    print(len(B4Mask), "b4masklength")
    print(len(B5Mask), "b5masklength")
    print(len(B8Mask), "b8masklength")
    npB4Mask = np.asarray([B4Mask])
    npB5Mask = np.asarray([B5Mask])
    npB8Mask = np.asarray([B8Mask])

    return npB4Mask, npB5Mask, npB8Mask


def GetNDVI(B8NIR, B4Red):
    NDVI = np.where(
    (B8NIR+B4Red)==0., 
    0,
    (B8NIR-B4Red)/(B8NIR+B4Red))   
    
    return NDVI

def GetNDRE(B8NIR, B5RE):
    NDRE = np.where(
        (B8NIR+B5RE)==0.,
        0.,
        (B8NIR-B5RE)/(B8NIR+B5RE)
    ) 
    return NDRE
def GetRECI(B5RE, B8NIR):
    RECI = np.where(
        B5RE==0.,
        0.,
        (B8NIR/B5RE) - 1
    )
    return RECI
def GetCCCI(B8NIR, B4Red, B5RE):
    CCCI = np.where(
        (B8NIR+B4Red)*(B8NIR+B5RE)==0.,
        0.,
        ((B8NIR - B5RE)/(B8NIR+B5RE)) / ((B8NIR-B4Red)/(B8NIR+B4Red))    
    )
    return CCCI

def GetCCCInew(NDVI, NDRE):
    newCCCI = np.where(
        NDVI==0.0,
        0.,
        NDRE/NDVI
    )
    return newCCCI

def GetIndices(B4Red, B5RE, B8NIR):
    NDVI = GetNDVI(B8NIR, B4Red)
    NDRE = GetNDRE(B8NIR, B5RE)
    RECI = GetRECI(B5RE, B8NIR)
    CCCI = GetCCCI(B8NIR, B4Red, B5RE)
    return NDVI, NDRE, RECI, CCCI

def GetIndicesnew(B4Red, B5RE, B8NIR):
    NDVI = GetNDVI(B8NIR, B4Red)
    NDRE = GetNDRE(B8NIR, B5RE)
    RECI = GetRECI(B5RE, B8NIR)
    CCCI = GetCCCInew(NDVI, NDRE)
    return NDVI, NDRE, RECI, CCCI
    
    return NDVI, NDRE, RECI, CCCI
def GetMaskedIndices(B4Red, B5RE, B8NIR):
    NDVI = GetNDVI(B8NIR, B4Red)
    NDRE = GetNDRE(B8NIR, B5RE)
    RECI = GetRECI(B5RE, B8NIR)
    CCCI = GetCCCI(B8NIR, B4Red, B5RE)
    
    NDVI = np.ma.masked_equal(NDVI, 0)
    NDRE = np.ma.masked_equal(NDRE, 0)
    RECI = np.ma.masked_equal(RECI, 0)
    CCCI = np.ma.masked_equal(CCCI, 0)

    
    return NDVI, NDRE, RECI, CCCI

def MakeMainDir(datadir, inputname, datatype='train'):
    newpath = os.path.join(datadir, datatype, inputname)
    os.mkdir(newpath)
    return newpath
def savedata(ndvi, ndre, reci, ccci, band4, datepath):
    ndvipath = os.path.join(datepath, 'NDVI.npz')
    ndrepath = os.path.join(datepath,'NDRE.npz')
    recipath = os.path.join(datepath, 'RECI.npz')
    cccipath = os.path.join(datepath, 'CCCI.npz')
    savez_compressed(ndvipath, ndvi)
    savez_compressed(ndrepath, ndre)
    savez_compressed(recipath, reci)
    savez_compressed(cccipath, ccci)
    SaveInfo(band4, datepath)

def savedatapng(ndvi, ndre, reci, ccci, band4, datepath):
   
    
    ndvipath = os.path.join(datepath, 'NDVI.png')
    ndrepath = os.path.join(datepath,'NDRE.png')
    recipath = os.path.join(datepath, 'RECI.png')
    cccipath = os.path.join(datepath, 'CCCI.png')

    matplotlib.image.imsave(ndvipath, ndvi)
    matplotlib.image.imsave(ndrepath, ndre)
    matplotlib.image.imsave(recipath, reci)
    matplotlib.image.imsave(cccipath, ccci)
    SaveInfo(band4, datepath)

    # Create bundle with save info
def SaveInfo(B4, datepath):
    si = dict()
    si['crs'] = B4.crs
    si['transform'] = B4.transform
    si['bounds'] = B4.bounds
    si['width'] = B4.width
    si['height'] = B4.height
    si['files']= B4.files
    si['name']= B4.name
    si['colorinterp']= B4.colorinterp
    si['compression']= B4.compression
    si['count']= B4.count
    si['dtypes'] = B4.dtypes
    sipath = os.path.join(datepath, 'SAVEINFO.csv')
    w = csv.writer(open(sipath, "w"))
    for key, val in si.items():
        w.writerow([key, val])
    
def MakeDatePath(raster, MainDirectory, SegList, i):
    rtail = GetTail(raster)
    rtail = RemoveExt(rtail)
    countypath = os.path.join(MainDirectory, rtail)
    
    D1 = SegList[i][0]
    print(D1)
    dsuf = GetTail(D1)
    
    DSufpre = dsuf.split('_B0')
    Datesuffix = DSufpre[0]

    DPath = os.path.join(countypath, Datesuffix)
    os.mkdir(DPath) # this create a datepath inside of countypath
    return DPath
    
def QualityCheck(MainDirectory):
    #iterate through directory
    for cpath in os.listdir(MainDirectory):
        for dpath in os.listdir(cpath):
            for filename in os.listdir(dpath):
                print(cpath, filename)
                temppath = os.path.join(dpath, filename)
                tempload = load(temppath)
                temparray = tempload['arr_0']
                plot.show(temparray, cmap='RdYlGn')
   
def RemoveExt(path):
    sp = os.path.splitext(path)
    newpath = sp[0]
    return newpath

def GetTail(path):
    headtail = os.path.split(path)
    return headtail[1]
    



if __name__ == "__main__":
    main()


# EXAMPLES: (6669, 10557) croppixels
#(6630, 7724) croppixels
#(6156, 9960) croppixels
#(6661, 7749) croppixels
#(6387, 9926) croppixels
#(6519, 10539) croppixels
#(5878, 9964) croppixels
#(6341, 10099) croppixels