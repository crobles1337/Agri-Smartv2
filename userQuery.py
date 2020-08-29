from userQueryhelpers import *
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
import geopandas as gpd
from datetime import date
from SentinelLoadstutzv2 import *
import logging




def userquery(input, ftype, crop, whistory = 240, predictions=True, flush=False, userdir = 'rtest/'):
    '''

    "Takes a tif, tiff, jp2, or shp file or coordinates as input from user. ftype is the type of input. If hweather is true, historical weather data will be saved. whistory is the number of days of weather history gathered. histndvi determines if historical ndvi is given. userdir is the directory for files to be saved for the user.   "
    
    Parameters:
    input(str): Path to either an input file, or a string in a wkt format of EPSG:4362 coordinates (ie. ()())
    ftype(str): A string denoting the input type as either 'coordinates', 'polygon', 'shp', 'jp2', 'tif', 'tiff'.
    crop(str): String of the crop name in first letter capitalized form (ie. 'Corn' or 'Wheat')
    whistory(int): An int denoting number of days of historical weather to access.
    predictions(bool): If true, crop yield and crop stage predictions will be produced if available (currently only for corn/wheat)
    flush(bool): If true, all jp2 satellite images downloaded in this function will be deleted immediately after.
    userdir(str): A str denoting the user directory all files will be saved within.

    Returns:
    True:

    Saves:
    32 jp2's


    '''
    predset = set(['corn', 'wheat'])
    if crop.lower() in predset:
        logger.debug('{crop} in set of predictable crops'.format(crop=crop))
    else:
        predictions == False
        logger.debug('{crop} is NOT in set of predictable crops'.format(crop=crop))
    


    # instantiate sentinel download class
    print('check1')

    bb, poly, acres = GetCoordinates(input, ftype)
    center = [(bb[0]+bb[2])/2, ((bb[1]+bb[3])/2)]

    # current date, and date from whistory number of days in the past
    start, end = datehistory(whistory)
    #s, e = datehistory(197) # is this no longer needed?

    username = 'croblitos' 
    pword = 'LucklessMonkey$30'
    sl = Sentinel2Loaderv2('', 
                    username, pword,
                    apiUrl='https://scihub.copernicus.eu/apihub/', showProgressbars=True, dateToleranceDays=10, loglevel=logging.DEBUG, savepath=userdir)

    # creates save paths depending on if a user directory is specified
    pathlist = makepathlist(userdir) 

    if userdir == None:
        hwpath = 'histweather.csv'
        fwpath = 'forecast.csv'
    else:
        #hndvipath = os.path.join(userdir, end + 'historicalndvi.png')
        hwpath = os.path.join(userdir, end + 'historicalweather.csv')
        fwpath = os.path.join(userdir, end + 'forecast.csv')
    
    # saves historical weather including cold/heat stress events to hwpath
    
    hwflat, rstats = GetHistWeather(center, whistory, path = hwpath)
    # saves 7 day weather forecast to fwpath
    GetWForecast(center[0], center[1], path=fwpath)


    fpaths = getbands(sl, poly, end, predictions=predictions) # THIS SHOULD GET ALL BANDS

    # saves bands at bandpaths
    cypaths, cspaths = getIndices(fpaths, pathlist, bb, userdir) # THIS SHOULD GET ALL IMAGES
    "cypaths, cspaths, maindict = getIndices(fpaths, pathlist, bb, userdir) # THIS SHOULD GET ALL IMAGES"

    #maindict will be appended to!

    if predictions == True:
        

        if crop.lower() == 'corn':
            csinput = getcstinput(cspaths)
            print('check9', len(csinput))
            pstage = predictstage(csinput, crop=crop)
            print('check10', pstage)    
        else:
            logger.debug('Crop Stage predictions only available  for corn.')
    
        cyinput = getcyinput(cypaths)
        pyield = predictyield(cyinput, crop=crop)               
        print('check12', pyield)    
    else:
        logger.debug('Crop stage and yield predictive analytics not available for {crop}'.format(crop=crop))
    if flush == True:
        flushpaths(None)

        "I may want to add a download True Color at the end, rn its buggy"

    return True #WILL NOW RETURN A DICTOONARY OF PATHS : STANDARDDICTIONARY

coo =  '37.5555, -120.2456, 37.5555, -120.1453, 37.6799, -120.1453, 37.6799, -120.2456, 37.5555, -120.2456'



inp = Polygon(((37.5555, -120.2456), (37.5555, -120.1453), (37.6799, -120.1453), (37.6799, -120.2456), (37.5555, -120.2456)))

inp2 = Polygon(((37.7130, -120.8656), (37.7130, -120.7137), (37.8659, -120.7137), (37.8659, -120.8656), (37.7130, -120.8656)))

userquery(inp2, 'polygon', 'Corn', 240, userdir='UQpractice/')








