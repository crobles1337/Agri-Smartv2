import requests
import datetime
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date, datetime, timedelta
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
import pickle
import csv
from datetime import date
from SentinelLoadstutzv2 import *
from indicesformulas import*
from MLhelpers import*
import math

# just prints week weather forecast
def GetWForecast(lat, lon, path, tempunit = 'metric'):
    '''
    Gets 7 day forecast 
    '''
    Days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    key = '35e02b4dab539973840fc771425f3539'

    WeatherStats = WeekForecast(lat, lon, key, tempunit)    
    dayofweek = datetime.now().weekday()
    week = list()
    week.append('Weather Forecast')
    for i in range(7):
        dayofweek = Days[(datetime.now().weekday() + i)%7]
        week.append(dayofweek)
    week.append('Units')
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([week])
        for k in WeatherStats:
            writer.writerow([k] + ["," + str(v) for v in WeatherStats[k]])
        


def WeekForecast(lat, lon, key, tempunit):
    '''

    '''
    exclude = 'minutely, current, hourly'
    WeatherData = requests.get('https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&units={}&exclude={}&appid={}'.format(lat, lon, tempunit,exclude, key))
    Weather = WeatherData.json()
    DailyW = Weather['daily']
    if tempunit == 'metric':
        TempUnit = 'celsius'
    Rain = [0]*8
    Rain[7] = 'mm'
    MinTemp = [0]*8
    MinTemp[7] = TempUnit
    MaxTemp = [0]*8
    MaxTemp[7] = TempUnit
    Humid = [0]*8
    Humid[7] = '%'
    DewPoint = [0]*8
    DewPoint[7] = TempUnit
    Clouds = [0]*8
    Clouds[7] = '%'
    Descrip = [0]*8
    Descrip[7] = ''
    UVIs = [0]*8
    UVIs[7] = ''
    Morn = [0]*8
    Morn[7] = TempUnit
    Night = [0]*8
    Night[7] = TempUnit
    Day = [0]*8
    Day[7] = TempUnit
    WindSpeed = [0]*8
    WindSpeed[7] = 'meter/sec'
    WindDeg = [0]*8
    WindDeg[7] = 'degrees'
    Eve = [0]*8
    Eve[7] = TempUnit

    for i in range(7):
        Day[i] = str(DailyW[i]['temp']['day'])
        Morn[i] = DailyW[i]['temp']['morn']
        Eve[i] = DailyW[i]['temp']['eve']
        Night[i] = DailyW[i]['temp']['night']
        MaxTemp[i] = DailyW[i]['temp']['max']
        MinTemp[i] = DailyW[i]['temp']['min']
        Humid[i] = DailyW[i]['humidity']
        DewPoint[i] = DailyW[i]['dew_point']
        WindSpeed[i] = DailyW[i]['wind_speed']
        WindDeg[i] = DailyW[i]['wind_deg']
        Descrip[i] = DailyW[i]['weather'][0]['description']
        UVIs[i] = DailyW[i]['uvi']
        Clouds[i] = DailyW[i]['clouds']

        if 'rain' in DailyW[i]:
            print("Rain: ", DailyW[i]['rain'], "mm")
            Rain[i] = DailyW[i]['rain']
    WeatherStats = dict({'Morning': Morn, 'Day': Day, 'Evening': Eve, 'Night': Night, 'Minimum Temperature': MinTemp, 'Max Temperature':MaxTemp, 'Humidity':Humid, 'Dew Point': DewPoint,'UVI': UVIs,'Cloud Coverage(%)': Clouds, 'Rain': Rain, 'Wind Speed': WindSpeed, 'Wind Degrees': WindDeg, 'Description':Descrip}) #

   
    return WeatherStats



def GetStationString(LatLon, MyAlt):
    '''

    '''
    stationlist = list()
    with open("CSVFiles\stationsAlt.pkl", "rb") as stationdict:
        stationdicts = pickle.load(stationdict)
        for k in stationdicts:
            if CalcDist(LatLon, stationdicts[k])<0.20:
                if AltDif(MyAlt, stationdicts[k])<450:
                    stationlist.append(k)
                    if len(stationlist)>48: #CHANGE IT BACK TO 48!!!!
                        break
        if len(stationlist)<15:
            for k in stationdicts:
                if CalcDist(LatLon, stationdicts[k])<0.45:
                    if AltDif(MyAlt, stationdicts[k]) < 400:
                        stationlist.append(k)
                        if len(stationlist)>48: #CHANGE IT BACK TO 48!!!!!
                            break
    print(len(stationlist), "lenstationlist")
    stationstring = ",".join(stationlist)
    return stationstring


def GetAltitude(LatLon):
    '''
    Gets altitude from coordinate using mapquest API.
    '''
    LatLonString = str(LatLon[0]) + ','+ str(LatLon[1])
    # Access altitude from coordinates
    AltKey = 'P9JZGGsBywA7Sx2IceReALUSOjsGQ8XQ'
    AltURL = 'http://open.mapquestapi.com/elevation/v1/profile?key={AltKey}&shapeFormat=raw&latLngCollection={LatLonString}'.format(AltKey=AltKey, LatLonString=LatLonString)
    MyAlt = requests.get(AltURL)
    MyAlt = MyAlt.json()
    if (MyAlt['info']['statuscode']==0):
        Altitude = MyAlt['elevationProfile'][0]['height']
    else:
        Altitude = -10000
    return Altitude

# takes two float tuples, calculates euclidian distance
def CalcDist(LatLon, v):
    '''
    Calculates euclidian distance between two coordinate points.
    
    Parameters:
    LatLon(tuple): Tuple of latitude,longitude
    v(dict item): A dictionary item containing station coordinate data.

    Returns:
    dist(float): Euclidian distance between desired location and station.
    '''
    xb = float(v[1])
    yb = float(v[0])
    dist =  ((LatLon[0] - xb)**2 + (LatLon[1]-yb)**2)**(1/2)

    return dist


# calculates difference in altitude (they are both in meters)
def AltDif(MyAlt, v):
    '''
    Calculates difference in altitude between a desired altitude and a station altitude.

    Parameters:
    MyAlt(float): A desired altitude.
    v(dict item): An item from a dictionary containing station altitudes. 

    Returns:
    AltDif(float): Difference in altitude.
    '''
    stationAlt = float(v[2])
    #check if either is missing value
    if ((MyAlt == -10000) or (stationAlt==-999)):
        return 0
    AltDif = abs(stationAlt - MyAlt)
    return AltDif


def GetHW(WS):
    '''
    Extracts individual weather stats as dictionaries from a json object from NOAA weather.

    Parameters:
    WS(json): A json object from the NCEI NOAA API containing weather stats.

    Returns:
    prcpDict, tmaxDict, tminDict, snowDict, snwdDict (dicts): 5 dictionaries of the various weather statistics. Dictionaries are formatted as date: (value, collisions) where collisions are the number of weather stations that provide the same location/date information to be averaged in order to fill weather data.


    '''
    # Dicts to save dated values
    newWS = WS
    prcpDict = dict()
    tminDict = dict()
    snowDict = dict()
    tmaxDict = dict()
    snwdDict = dict()

    for i in range(len(newWS)):
        if 'PRCP' in newWS[i]:
            MMPrecip = float(newWS[i]['PRCP'])/10
            #if date already stored, average over values
            if newWS[i]['DATE'] in prcpDict:
                Collisions = prcpDict[newWS[i]['DATE']][1] + 1
                # date's value = new value* 1/collisions + old value* (1 - 1/collisions) 
                prcpDict[newWS[i]['DATE']] = ((MMPrecip*(1/Collisions)) + (prcpDict[newWS[i]['DATE']][0]*(1-(1/Collisions))), Collisions)
            else:    
                prcpDict[newWS[i]['DATE']] = (float(newWS[i]['PRCP']), 1)
    
        if 'TMIN' in newWS[i]:
            CelsiusTMIN = float(newWS[i]['TMIN'])/10
            if newWS[i]['DATE'] in tminDict:
                TMINCollisions = tminDict[newWS[i]['DATE']][1] + 1
                tminDict[newWS[i]['DATE']] = ((CelsiusTMIN*(1/TMINCollisions))+(tminDict[newWS[i]['DATE']][0]*(1-(1/TMINCollisions))), TMINCollisions)    
        
            else:
                tminDict[newWS[i]['DATE']] = (CelsiusTMIN, 1)
    
        if 'SNOW' in newWS[i]:
            if newWS[i]['DATE'] in snowDict:
                SNOWCollisions = snowDict[newWS[i]['DATE']][1] + 1
                snowDict[newWS[i]['DATE']] = ((float(newWS[i]['SNOW'])*(1/SNOWCollisions))+(snowDict[newWS[i]['DATE']][0]*(1-(1/SNOWCollisions))), SNOWCollisions)
            else:
                snowDict[newWS[i]['DATE']] = (float(newWS[i]['SNOW']), 1)
    
        if 'TMAX' in newWS[i]:
            CelsiusTMAX = float(newWS[i]['TMAX'])/10
            if newWS[i]['DATE'] in tmaxDict:
                TMAXCollisions = tmaxDict[newWS[i]['DATE']][1] + 1 
                tmaxDict[newWS[i]['DATE']] = ((CelsiusTMAX*(1/TMAXCollisions))+(tmaxDict[newWS[i]['DATE']][0]*(1-(1/TMAXCollisions))), TMAXCollisions)
                
            else:
                tmaxDict[newWS[i]['DATE']] = (CelsiusTMAX, 1)
                

        if 'SNWD' in newWS[i]:
            if newWS[i]['DATE'] in snwdDict:
                SNWDCollisions = snwdDict[newWS[i]['DATE']][1] + 1
                snwdDict[newWS[i]['DATE']] = ((float(newWS[i]['SNOW'])*(1/SNWDCollisions)) + (snwdDict[newWS[i]['DATE']][0]*(1-(1/SNWDCollisions))), SNWDCollisions)
    return prcpDict, tmaxDict, tminDict, snowDict, snwdDict, 


def HeatStressCheck(croptype, tmax):
    "Identifies heat stress events to be charted from historical temperature. "
    HeatStressDict = {'corn': 35, 'avocado': 35, 'wheat': 32, 'sugarcane': 40, 'jalapeno': 32
    }
    stress = HeatStressDict[croptype]
    events = dict()
    for item in tmax:
        if tmax[item]>=stress: # [0] subscript needed if collisions not removed
            events[item] = tmax[item] # [0] subscript needed if collisions not removed
    count = len(events)
    return events, count
    

def ColdStressCheck(croptype, tmin):
    "Identifies cold stress events to be charted from historical temperature. "
    ColdStressDict = {'corn': 8, 'avocado': 0, 'wheat': 9, 'sugarcane': 18, 'peppers': 12 
    }
    stress = ColdStressDict[croptype]
    events = dict()
    for item in tmin:
        if tmin[item]<=stress: # [0] subscript needed if collisions not removed
            events[item] = tmin[item] # [0] subscript needed if collisions not removed
    count = len(events)
    return events, count


def RemoveCollisions(stats):
    "Removes collision values (previously used to average weather values) from historical weather. Deprecated as weather is no longer used as a parameter for machine learning algorithm."
    # stats is list of dicts
    for d in stats:
        for v in d.keys():
            d[v] = d[v][0]
    return stats


def GetHistWeather(lonlat, whistory, path, crop = 'wheat'):
    '''
    Gets historical weather from the noaa global weather station database.

    Parameters:
    lonlat(): A tuple of longitude and latitude values.
    whistory(int): The number of days of history to be retrieved.
    path(str): Path where historical weather will be saved.
    crop(str): Crop name is string format (ie. 'Corn')

    Returns:
    hwflat(array): An array of all historical weather flattened (deprecated usage as ML parameter)
    rstats(array): A list of cold stress and heat stress events.


    Saves:
    A csv of historical weather data including precipitation(mm), max/min temperature(C), and snow(mm) and cold/heat stress events.

    '''

    "Latlon = tuple of latitude and longitude values. start Saves historical weather including precipitation, max temp, min temp, snow, snow depth, cold and heat stress events at coordinate from START to END dates, at the path specified. Cold/Heat Stress available for 'wheat, avocado, corn, sugarcane, and jalapeno. Auto-setting gives stress events for wheat. Returns no values."
    
    if whistory > 197:
        start, end = datehistory(whistory)
        dif = whistory - 197
    else:
        start, end = datehistory(197)
        whistory = 197

    # format inputs to weather API
    LatLonString = str(lonlat[1]) + ','+ str(lonlat[0]) # b/c center is lonlat format
    LatLon = [lonlat[1], lonlat[0]]
    MyAlt = GetAltitude(LatLon) # unit = meters
    stationstring = GetStationString(LatLon, MyAlt)
    dates = 'startDate={start}&endDate={end}'.format(start=start, end=end)
    WeatherStats = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations={stationstring}8&dataTypes=SNOW,PRCP,TMIN,TMAX,SNWD&{dates}&includeAttributes=true&includeStationName:1&includeStationLocation:1&format=json'.format(stationstring=stationstring, dates = dates))
    
    # format API output, average weather values, and remove repeats for raw stats displayed to user
    newWS = WeatherStats.json()
    prcpDict, tmaxDict, tminDict, snowDict, snwdDict = GetHW(newWS)
    rstats = [prcpDict, tmaxDict, tminDict, snowDict, snwdDict]
    rstats = RemoveCollisions(rstats)

    # get temperature stress stats
    cs, cscount, = ColdStressCheck(crop, tminDict)
    hs, hscount = HeatStressCheck(crop, tmax=tmaxDict)
    rstats.append(cs)
    rstats.append(hs)

    # impute missing values for machine learning input
    yearDict = GetDateDict(whistory) 
    impyeardict = GetDateDict(197) # 197 was a standard I used for building the training data
    # check if lr matches up by date so that I can cut auto cut off at 197 by using a different year dict
    ImpPrcp = dict()
    ImpTmin = dict()
    ImpTmax = dict()
    ImpSnow = dict()
    ImpSnwd = dict()
    ImpPrcp = GetLR(impyeardict, prcpDict, ImpPrcp)
    ImpTmin = GetLR(impyeardict, tminDict, ImpTmin)
    ImpTmax = GetLR(impyeardict, tmaxDict, ImpTmax)
    ImpSnow = GetLR(impyeardict, snowDict, ImpSnow)
    ImpSnwd = GetLR(impyeardict, snwdDict, ImpSnwd)
    stats = [ImpPrcp, ImpTmax, ImpTmin, ImpSnow, ImpSnwd]
    hwflat = []
    # save stats with predicted values for ML input
    [hwflat.extend(list(stat.values())) for stat in stats]
    hwflat.append(cscount)
    hwflat.append(hscount)

    # save raw weather as csv to path
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', ',Precipitation (mm)', ',Max Temp (C)', ',Min Temp (C)', ',Snowfall (mm)', ',Snow Depth(mm)', ",Cold Stress Events", ",Heat Stress Events"])
        for key in sorted(yearDict.keys(), key=lambda x: x):
            writer.writerow([key] + ["," + str(d.get(key, None)) for d in rstats])

    return hwflat, rstats 


def Zoning(nparr, indextype):
    '''
    Zones a region into 5 different percentile values.

    Parameters: 
    nparr: An index to be zoned. 
    indextype(str): The vegetation index as a string such as 'ndvi'
    
    Returns: 
    npzone[0] (np array): An np array of the same index but zoned by percentiles of values, and the cutoff values for the given index.
    zones(arr): An array containing percentile value cutoffs to be used as a legend to understand values. 
    '''
    if indextype.lower() in ['ndvi', 'ndre', 'sipi', 'ndmi', 'gvmi', 'mi']: # list of indices going from -1 to 1
        mx = 1
        mn = -1
    if indextype.lower() in ['reci', 'gci', 'arvi', 'savi']: # list of indices going really high
        mx = 100
        mn = -100
    # clip values
    nparr = np.where(
        nparr<mn,
        mn,
        nparr
    )
    nparr = np.where(
        nparr>mx,
        mx,
        nparr
    )
    maskedarr = np.ma.masked_where(nparr==0, nparr)
    npzone = np.copy(maskedarr)
    nanarr = np.ma.filled(maskedarr, np.nan)

    zones = [np.min(maskedarr), np.nanpercentile(nanarr, 20), np.nanpercentile(nanarr, 40), np.nanpercentile(nanarr, 60), np.nanpercentile(nanarr, 80), np.max(maskedarr)]
    
    print(npzone.shape, "npzoneshape")
    for i in range(len(zones)-1):
        top = zones[i+1]
        bot = zones[i]
        npzone = np.where(
            (bot < npzone) &  (npzone < top),
            zones[i+1],
            npzone
        )
    print(npzone.shape, "npzoneshape2")
    print(npzone[0].shape, "npzoneshape3")
    return npzone[0], zones # returns zoned index, as well as cutoff values


def banddelete(bplist):
    None


def bandtonpz(bpblist):
    "takes in a list of band paths, saves all as npz's and then deletes raw bands. Returns void"
    nplist = [p.replace('jp2', 'npz') for p in bplist]
    # open the bands
    for i in range(bplist):
        tmp = rasterio.open(bplist[i])
        band = tmp.read(1)
        savez_compressed(nplist[i], band)
        band.close()
    [os.remove(bp) for bp in bplist]


def historicalndvi(bandpaths, bb):
    "Gets ndvi images going back 180 days. sl = SentinelLoader object. coordinate = Polygon or point to be transformed to wkt for sentinel load query. Count = number of images to be used in historic info. Returns hndvi which is the averaged ndvi image over past count* ndvi images, and a list of np arrays with data from ndvi images"

    # get current date
    today = date.today()
    
    dlist = []
    b4plist = []
    b8plist = []
    nlist = []
    nirplist = []

    avgs = []
    for i in range(count):
        delta = i * 20
        day = append(today - timedelta(days = delta))
        dlist.append(day.strftime("%Y-%m-%d"))
    for date in dlist:
        b4plist.append(sl.getProductBandTiles(coordinate, 'b04', '10m', date))
        b8plist.append(sl.getProductBandTiles(coordinate, 'b08', '10m', date))
        
 #       nirplist.append(sdownload(date, 'b08', '10m')) #ognirbands
 #       redplist.append(sdownload(date, 'b04', '10m')) #ogredbands
    # save all historical ndvi for viewing
    #window = windowfunction(bb)

    for i in range(len(bandpaths)):
        if 'B04' in bandpaths[i]:
            b4path = bandpaths[i]
            b8path = bandpaths[i].replace('B04', 'B08')
            
            b4 = rasterio.open(b4path, driver = 'JP2OpenJPEG')
            b8 = rasterio.open(b4path, driver = 'JP2OpenJPEG')
            window10m, window20m, window100m = genWindow(b4, bb)
            
            redlist.append(b4.read(1, window=window10m).astype('float64'))
            nirlist.append(b8.read(1, window=window10m).astype('float64'))


    for i in range(10):
        nlist[i] = getNDVI(redlist[i], nirlist[i])

    ndvis = np.array(nlist)
    ndvis.sum(axis=0)
    hndvi = ndvis/len(nlist)

    return hndvi, nlist
    

def GetCoordinates(gfile, ftype, n = 1, acrlim = 10000000):
    '''
    Gets coordinates from valid path or set of coordinates as a comma separated string.

    Parameters:
    gfile=This.
    gfile(str): Either a path to a valid .tif, .jp2, or .shp file w/ coordinates available, or a comma seperated string of coordinates such as '37, -120, 38, -120, 38, -121, 37, -121, 37, -120'
    ftype(str): The type of file as either 1. 'shp' 2. 'tif' 3. 'jp2' 4. 'coordinates'
    acrlim(int): A size limit in acres for the space allowed.

    '''


    "Gets coordinates from valid file. gfile is a tif, jp2, or shape file w/ coordinates available. ftype is the file type. n is an optional parameter, defining the number of shapes in shape file if there are more than one. Lastly, acrlim is the acreage limit on a request, ensuring the "
    n = 1
    if ftype == 'shp':
        f = gpd.read_file(gfile)
        f = f.to_crs(epsg=4326)
        shplist = list()
        bblist = list()
        for i in range(n):
            shplist.append(f['geometry'][i])
            bblist.append(f['geometry'][i].bounds)
        
        sizelist = list()  
        [sizelist.append(checkSize(bb, acrlim)[0]) for bb in bblist]
        if sum(sizelist)<acrlim:
            acr = sum(sizelist)
            return shplist, acr
        else:
            raise Exception('Total acreage exceeds acreage limit with ', acr, 'acres')
        
    if ftype == ('tif' or 'jp2' or 'tiff'):
        with rasterio.open(gfile) as f:
            bounds = f.bounds
            print(bounds)
            print(len(bounds))
            xs = [bounds[0], bounds[2]]
            ys = [bounds[1], bounds[3]]
            xy = rasterio.warp.transform(
                src_crs=f.crs, dst_crs='EPSG:4326', xs = xs, ys = ys)
            [print(c[0], c[1]) for c in xy]
            bb = [xy[0][0], xy[1][0], xy[0][1], xy[1][1]]
    
    if ftype == 'polygon':
        poly = gfile
        bb = poly.bounds

    if ftype == 'coordinates':
        bb, poly = Coordinatesfromstring(gfile)
    
    if ftype == 'wkt':
        bb = gfile.bounds
    
    acr, check = checkSize(bb, acrlim)
    if check == False:
        raise Exception('Total acreage exceeds acreage limit with ', acr, 'acres')
        return 0
    poly = Polygon([(bb[1], bb[0]), (bb[3], bb[0]), (bb[3], bb[2]), (bb[1], bb[2]), (bb[1], bb[0])]) 

    return bb, poly, acr

def Coordinatesfromstring(coos):
    '''
    Converts a comma seperated string of coordinates into a polygon item and a bounding box object. 
    
    Parameters:
    coos(str): A string of comma seperated coordinates forming a polygon. Ie. '37.5555, -120.2456, 37.5555, -120.1453, 37.6799, -120.1453, 37.6799, -120.2456, 37.5555, -120.2456'. Automatically closes polygon if open.

    Returns:
    bb(bounding box): A bounding box object bounding the coordinates in the polygon.
    poly(Polygon): A polygon object matching the inputted coordinates.
    '''
    co = coos.split(',')
    if len(co)%2!=0:
        raise Exception("Coordinate pairs incomplete!")
    clist = []
    cp = []
    [clist.append(float(c)) for c in co]
    for i in range(len(coo2)):
        if i%2==1:
            None
        else:
            cp.append((cos2[i], cos2[i+1]))
    # Check if polygon is closed
    if cp[0]!= cp[len(cp)]:
        cp.append(cp[0])
    poly = Polygon(cp)
    bb = poly.bounds
    return bb, poly




def checkSize(bb, acrlim):
    "takes bounding box and returns acreage, and true if less than acreage limit"
    #54.6 is 1 longitude
    #69 is 1 latitude

    acr = (((bb[2] - bb[0])*69.0000) * ((bb[3]-bb[1])*54.6000))/0.0015625
    if acr<acrlim:
        check = True
    else:
        check = False
    return acr, check


def datehistory(delta):
    '''
    Get the current date, and the date from a specified number of days ago.
    
    Parameters:
    delta(int): An integer denoting number of days ago to find.

    Returns:
    past(str): A string in the format 'YYYY-mm-dd' of the date that is delta days ago.
    today(str): A string in the format 'YYYY-mm-dd' of the current date.

    '''
    today = date.today()
    past = today - timedelta(days = delta)
    today = today.strftime("%Y-%m-%d")
    past = past.strftime("%Y-%m-%d")
    return past, today


def makepathlist(dirpath = None):
    "Creates list of paths to place vegetation at the selected directory path for the 10 current day indices!!!"
    today = date.today()
    today = today.strftime("%Y_%m_%d_")
    pathlist = list()
    indexlist = ['ndvi', 'ndre', 'reci', 'savi', 'sipi', 'arvi', 'gvmi', 'ndmi', 'gci', 'ndwi', 'mi']
    if dirpath==None:        
        [pathlist.append(today + v + '.png') for v in indexlist]
    else:
        [pathlist.append(os.path.join(dirpath, (today + v + '.png'))) for v in indexlist]
    
    return pathlist

def getcompositebands(sl, poly, dat, predictions):
    '''
    Gets composite bands in the case that no singular image covers the entire desired area.

    Returns:
    fpaths(): paths to the new, composite images created
    '''
    # Downloaad all images include in sl w/ getproductbandtilescomposite function
        # this function should download all, and the use crop region
    # Return output from crop region


# gets bands 
def getbands(sl, poly, dat, predictions):
    '''
    Gets current image bands of plot, and returns paths to all saved jp2 files.
    
    Parameters:
    sl(SentinelLoaderv2): A sentinel loader object used to download bands.
    poly(Polygon): A polygon object with the coordinates of the specified region.
    dat(date): A date object of the current date.
    predictions(bool): If true, will save bands necessary for crop yield/crop stage predictions.

    Returns:
    fpaths(list): A list of strings containing paths to all downloaded bands as jp2's

    '''
    date2 = datetime.strptime(dat, "%Y-%m-%d")
    # download relevant bands from current date
    blist = []
    indexlist = []
    b4dtlist = [] 

    blist = [('B02', '10m'), ('B03', '10m'), ('B04', '10m'), ('B05', '20m'),('B08', '10m'),  ('B8A', '20m'), ('B11', '20m'), ('B12', '20m'), ('TCI', '10m') ]
    bdict = {'B02': '10m', 'B03': '10m', 'B04': '10m', 'B05': '20m', 'B08': '10m',  'B8A': '20m', 'B11': '20m', 'B12': '20m'} # tci was here before
    fpaths = list()

    

    #testpath = sl.getProductBandTiles(poly, 'B02' , bdict['B02'], dat)
    #testband = rasterio.open(testpath, driver = 'JP2OpenJPEG')
    #window10m, window20m, window100m, composite = genWindow(testband, bb)
    #if composite == True:
    #    fpaths = getcompositebands(sl, poly, dat, predictions)
    #    return fpaths, composite

    #(1, window=window10m).astype('float64')

    for k in bdict.keys():
        
        if k == 'B04':
            fpath = sl.getProductBandTiles(poly, k, bdict[k], dat)
            fpaths.append(fpath)
            b4dtlist.append(dtfrompath(fpath))
        else:

            fpaths.append(sl.getProductBandTiles(poly, k, bdict[k], dat))

    print(b4dtlist, "this is b4dtlist with a length of ",len(b4dtlist))

    # Collect historical bands for 1) histndvi 2) crop yield ML 3) crop stage ML
    if predictions == True:   
        dlist = []
        b4plist = []
        b8plist = []
        nlist = []
        nirplist = []
        avgs = []
        daylist = []
        # Creates list of historical dates
        for i in range(9):
            delta = (i+1) * 20 # i+1 to NOT include the current bands already downloaded
            day = date2 - timedelta(days = delta)
            daylist.append(day) # saved as date
            dlist.append(day.strftime("%Y-%m-%d")) # saved as d
         
        d8 = datetime.today()
        year = d8.year
        indexlist = []
        optimaldates = [datetime(year,7,1), datetime(year,6,10), datetime(year,5,15), datetime(year,4,25), datetime(year,4,1)]
        # get optimal dates for band 5
        b5dts = []
        for opt in optimaldates:
            best = 365
            for dy in daylist:
                delt = dy - opt
                if abs(delt.days) < best:
                    if not dy.strftime("%Y-%m-%d") in b5dts:
                        best = abs(delt.days)
                        b5dt = dy   
            b5dts.append(b5dt.strftime("%Y-%m-%d"))
        for dt in dlist:

            b4dt = (sl.getProductBandTiles(poly, 'B04', '10m', dt)) ## just changed all these bands to capitals. Formerly b04
            fpaths.append(b4dt)
            tmpdt = dtfrompath(b4dt) # returns a date object
            b4dtlist.append(tmpdt)
            fpaths.append(sl.getProductBandTiles(poly, 'B08', '10m', dt))
            print("fpaths finish ")

            if dt in b5dts:
                tmp = sl.getProductBandTiles(poly, 'B05', '20m', dt)
                if tmp in fpaths:
                    None
                    print("tmp was already in fpaths!")
                else:
                    fpaths.append(tmp)
                    print("appending {tmp} to fpaths!".format(tmp=tmp))

        # TOTALED
    # 10* b04 b08 over past 6 months. 10 will be displayed and averaged for historical ndvi. 5 from past 4 months will be used for CYinput. 1 current will be used for current indices and CS input.
    # 5 * b05 over past 4 months for  CY input. 1 current will be used for current indices and CS input
    # 1 * b02, b03, b08, b8A, b11, b12, and current indices
    # totals to 31 bands 
    return fpaths


def GetDateDict(delta):
    #days should be number of days inpast
    now = date.today()
    datedict = dict()
    
    for i in range(delta):
        newdate = now - timedelta(days = delta - i)
        newdate = newdate.strftime("%Y-%m-%d")
        datedict[newdate] = i + 1 # there is an offset and it starts at 1 not 0
    return datedict


def GetYearDict(year):
    '''
    Creates a dictionary pairing all dates in the year with how many days have passed in the year at the given date.

    Parameters:
    year(str): A string denoting the full year (ie. '2019') 

    Returns:
    yeardict(dict): A dictionary object with all dates in the given year, paired the number of days passed at the given date.
    '''

    dict2020 = dict()
    dict2019 = dict()
    dict2018 = dict()
    with open('CSVFiles/2020Dates.csv') as f:
        count = 0
    
        for w in f:
            
            wbase = w.split(',')
            
            #print(w2[0])
            w2020 = wbase[0]
            w2020 = w2020.replace("\t", '')
            w2020 = w2020.replace('"', '')
            w2019 = w2020.replace('2020', '2019')
            w2018 = w2020.replace('2020', '2018')
            #print(w3, 'w3')
            dict2018[w2018] = count
            dict2019[w2019] = count
            dict2020[w2020] = count 
            count = count+1
    if year == '2020':
        yeardict = dict2020
    if year == '2019':
        yeardict = dict2019
    if year == '2018':
        yeardict = dict2018
    return yeardict



def genWindow(band10m, bb):
    '''
    Generates appropriate windowing for a bounding box in EPSG:4326 for 10m, 20m, and 100m bands.

    Paramaters:
    band10m():
    bb():

    Returns:
    window10m: A window object containing the window to be indexed in a 10m band to show the desired plot of land.
    window20m: A window object containing the window to be indexed in a 20m band to show the desired plot of land.
    window100m: A window object containing the window to be indexed in a 100m band to show the desired plot of land.
    '''

    x, y = rasterio.warp.transform('EPSG:4326', band10m.crs, [bb[1], bb[3]], [bb[0], bb[2]])
    row, col = band10m.index(x, y)
    c1 = col[0]
    r1 = row[0]
    c2 = col[1]
    r2 = row[1]
    if c1<c2:
        coloff = c1
        cval = c2 - coloff
    else:
        coloff = c2
        cval = c1 - coloff
    if r1<r2:
        rowoff = r1
        rval = r2 - rowoff
    else:
        rowoff = r2
        rval = r1 - rowoff

    if (cval%2) != 0:
        cval = cval + 1
    if (rval%2) != 0:
        rval = rval + 1
    if (coloff%2) != 0:
        coloff = coloff + 1
    if (rowoff%2) != 0:
        rowoff = rowoff + 1

    composite = False
    for v in [coloff, rowoff, cval, rval]:
        if v<0:
            composite = True
            print("composite == True", coloff, rowoff, cval, rval)
            v=0

    window10m = Window(coloff, rowoff, cval, rval)
    window20m = Window(coloff/2, rowoff/2, cval/2, rval/2)
    window100m = Window(coloff/10, rowoff/10, (cval)/10, (rval)/10)

    return window10m, window20m, window100m, composite

def savestats(stat, indicator, userdir = '', itemname='', autopath=True):
    "Saves np array as an npz. Returns path."
    path = os.path.join(userdir, itemname, stat +'.npz')
    savez_compressed(path, statsfromindices(indicator))
    print(path)
    return path

def pngsave(name, data, userdir = '', itemname='', autopath=True):
    "Saves np array as png image. Name is the stat type (ndvi/ndre, etc.). data == np array. Userdir is the directory to be saved. itemname is the product name. If autopath== true, it is saved only as the filename, with no userdir/itemname specified. RETURNS: path to png."
    if autopath == True:
        path = os.path.join(userdir + itemname + name + '.png')
    else:
        path = name + '.png'
    matplotlib.image.imsave(path, data)
    print("New png image saved at", path)
    return path

def flushpaths(paths):
    '''
    Clears all data from the specified paths.
    '''
    for p in paths:
        if os.path.splitext(p)[1]=='.jp2':
            os.remove(p)
            logger.info('{p} has been removed'.format(p=p))

    return True

# collects indices from saved bands, 
def getIndices(fpaths, imsavepaths, bb, userdir):
    "fpaths: list of 9 bands from most recent satellite imagery. LatLon: tuple of EPSG:4326 coordinates. Returns: list of np arrays containing 11 spectral indices, NDVI, NDRE, RECI, SAVI, SIPI, ARVI, GVMI, NDMI, GCI, NDWI, MI. Saves directly to bandpaths"
    
    maindict = {'current_ndvi':'', 'current_ndre':'','current_reci':'', 'current_savi':'','current_sipi':'', 'current_arvi':'','current_gvmi':'', 'current_ndmi':'','current_gci':'', 'current_ndwi':'','current_mi':'', 'current_ndre':'','historical_ndvi':'', 'historical_ndvi_zoned':'' }
    # 

    "Figure out if imsavepaths are needed?"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"
    "REMEMBER TO RENAME ALL THE BANDS!!!!!!!!!!!"


    band2 = rasterio.open(fpaths[0], driver = 'JP2OpenJPEG') #blue - 10m
    band3 = rasterio.open(fpaths[1],  driver = 'JP2OpenJPEG') #green - 10m        
    band4 = rasterio.open(fpaths[2],  driver = 'JP2OpenJPEG') #red - 10m
    band5 = rasterio.open(fpaths[3],  driver = 'JP2OpenJPEG') #red edge close to center - 20m, 5990
    band8 = rasterio.open(fpaths[4],  driver = 'JP2OpenJPEG') #nir - 10m, 10980
    band8A = rasterio.open(fpaths[5],  driver = 'JP2OpenJPEG') #narrownir - 20m, 5990
    band11 = rasterio.open(fpaths[6],  driver = 'JP2OpenJPEG') #swir1 - 20m
    band12 = rasterio.open(fpaths[7],  driver = 'JP2OpenJPEG') #swir2 - 20m
    
    #bandTCI = rasterio.open(fpaths[8],  driver = 'JP2OpenJPEG') #true color image, 10m

    window10m, window20m, window100m, composite = genWindow(band4, bb)

    
    ogRedBand = band4.read(1, window=window10m).astype('float64')
    ogNIRBand = band8.read(1, window=window10m).astype('float64')
    rededge = band5.read(1, window=window20m).astype('float64')
    ogGreenBand = band3.read(1, window=window10m).astype('float64')
    ogBlueBand = band2.read(1, window=window10m).astype('float64')
    wnnir = band8A.read(1, window=window20m).astype('float64')
    wswir1 = band11.read(1, window=window20m).astype('float64')
    wswir2 = band12.read(1, window=window20m).astype('float64')

    #Stretch all 20m bands to 10980x10980
    ogRedEdgeStretch = np.repeat(np.repeat(rededge,2, axis=0), 2, axis=1) 
    ogSWIR1Stretch = np.repeat(np.repeat(wswir1,2, axis=0), 2, axis=1) 
    ogSWIR2Stretch = np.repeat(np.repeat(wswir2,2, axis=0), 2, axis=1) 
    ogNNIRStretch = np.repeat(np.repeat(wnnir,2, axis=0), 2, axis=1) 


    # Calculate NDVI
    nNDVI = getNDVI(ogRedBand, ogNIRBand)
    nNDRE = getNDRE(ogRedEdgeStretch, ogNIRBand)
    nRECI = getRECI(ogRedEdgeStretch, ogNIRBand)
    nSAVI = getSAVI(ogRedBand, ogNIRBand)
    nSIPI = getSIPI(ogBlueBand, ogRedBand, ogNIRBand)
    nARVI = getARVI(ogBlueBand, ogRedBand, ogNIRBand)
    nGVMI = getGVMI(ogNIRBand, ogSWIR1Stretch)
    nNDMI = getNDMI(ogNNIRStretch, ogSWIR1Stretch, ogSWIR2Stretch)
    nGCI = getGCI(ogGreenBand, ogNIRBand)
    nNDWI = getNDWI(ogNNIRStretch, ogSWIR1Stretch)
    nMI = getMI(ogNNIRStretch, ogSWIR2Stretch)
    
    pngpaths = []

    # save all current indices as pngs
    indices = [nNDVI, nNDRE, nRECI, nSAVI, nSIPI, nARVI, nGVMI, nNDMI, nGCI, nNDWI, nMI]

    [print(item.shape, "item") for item in indices]
    [pngpaths.append(pngsave(imsavepaths[i], indices[i], autopath=False)) for i in range(len(indices))]

    
    cspaths = []
    cspaths.append(savestats('ndrestatscurrent', nNDRE, userdir))
    cspaths.append(savestats('ndvistatscurrent', nNDVI, userdir))
    # historical ndvi indices
    rlist = []
    nirlist = []
    relist = []
    b4list = []
    b8list = []
    indexb4 = []
    indexb8 = []
    print(len(fpaths), "fpathslength")
    # open nirbands, match with redband

    "this works if all indices are in fpaths -- figure out if you fixed this simply by using .lower() on the b04/b08 checks, or if fpaths needs to have collected more paths from the getgo"
    for i in range(len(fpaths)):
        # collect ndvis
        if 'b08' in fpaths[i].lower():
            b = fpaths[i]
            c = b.replace('B08', 'B04')

            b4 = rasterio.open(b, driver = 'JP2OpenJPEG')
            b8 = rasterio.open(c, driver = 'JP2OpenJPEG')

            b4read = b4.read(1).astype('float64')
            b8read = b8.read(1).astype('float64')
            b4list.append(c)
            b8list.append(b)

            rlist.append(b4.read(1, window=window10m).astype('float64'))
            nirlist.append(b8.read(1, window=window10m).astype('float64'))
            b4.close()
            b8.close()
    b5len = 0
    for i in range(len(fpaths)):
        if 'b05' in fpaths[i].lower():
            b5len = b5len + 1
    if b5len != 5:
        ignore = fpaths[3]
    else:
        ignore = ' '



    for i in range(len(fpaths)):
        if 'b05' in fpaths[i].lower():
            if fpaths[i]!=ignore:
                b = fpaths[i]
                # collect index of rlist and nirlist to access bands
                c = b.replace('B05', 'B04')
                d = b.replace('B05', 'B08')
                indexb4.append(b4list.index(c))
                indexb8.append(b8list.index(d))
                b4 = rasterio.open(b, driver = 'JP2OpenJPEG')
                b5 = rasterio.open(c, driver = 'JP2OpenJPEG')
            
                b5read = b5.read(1).astype('float64')

                b5s = b5.read(1, window=window20m).astype('float64')
                b5stretch = np.repeat(np.repeat(rededge,2, axis=0), 2, axis=1) 
                relist.append(b5stretch)


    ndvilist = []
    ndrelist = []
    recilist = []
    cccilist = []
    cypaths = []
    print(len(nirlist), "length of nirlist")
    # place all 10 hist ndvi images in nlist
    recount = 0
    for i in range(len(nirlist)):
        itemname = os.path.splitext(b4list[i].replace('B04', ''))[0] 
        poop = rlist[i]
        pop = nirlist[i]
        ndvilist.append(getNDVI(rlist[i], nirlist[i]))
        
        pngpaths.append(pngsave('ndvi', ndvilist[i])) ############################################## userdir, itemname deleted!
        # if matches 5 red edge bands NOTE: Check if matches reverse chronological order
        print(len(indexb4), "length of indexb4")
        if i in indexb4: 
            print(type(relist[recount]), 'typerelist')
            

            NDRE = getNDRE(relist[recount], nirlist[i])
            RECI = getRECI(relist[recount], nirlist[i])
            CCCI = getCCCI(ndvilist[i], NDRE)
            recount = recount + 1  
            cypaths.append(savestats('cccistats{ind}'.format(ind=str(i)), CCCI))
            cypaths.append(savestats('ndrestats{ind}'.format(ind=str(i)), NDRE))
            cypaths.append(savestats('ndvistats{ind}'.format(ind=str(i)), ndvilist[i]))
            cypaths.append(savestats('recistats{ind}'.format(ind=str(i)), RECI))
            
            pngpaths.append(pngsave('ndre', NDRE, itemname = itemname)) # if doesn't work, add userdir and label as inputs
            pngpaths.append(pngsave('reci', RECI, itemname = itemname))
            pngpaths.append(pngsave('ccci', CCCI, itemname = itemname))
        # add zoning here if you would like                    

    print(cypaths, "THIS IS CYPATHS")
    ndvis = np.array(ndvilist)
    ndvis.sum(axis=0)
    hndvi = ndvis/len(ndvilist)
    hndvizoned, zones = Zoning(hndvi, 'ndvi') 
    tod = date.today().strftime("%Y-%m-%d")
    with open(userdir + tod +'HNDVIzoned.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([str(z) + ","  for z in zones])

    hndvipath = pngsave('HistoricalProductivityNDVI', hndvi[0], userdir, 'hist')
    hndvizpath = pngsave('HistoricalProductivityZonedNDVI', hndvizoned, userdir, 'hist')

    return cypaths, cspaths


    

def statsfromindices(indices):
    "Takes a single spectral band index (such as ndre, ndvi, etc.) and returns a list of statistical indicators to be used for ML input"
    stats = [np.average(indices), np.quantile(indices, .25), np.quantile(indices, .5), np.quantile(indices, .75)]
    return stats

def getcropstage():
    None


def getcropyield():
    None
    "applies trained ML to "
    "add in makebundle, "


# functions used for imputation

def GetLR(yeardict, statDict, Imp):
    "Uses linear regression to perform imputation on missing values. Returns new list with predicted values the size of yeardict."

    for key in yeardict.keys():
        if key in statDict:
            Imp[yeardict[key]] = statDict[key]

    Xl = list(Imp.keys())
    X = np.array(Xl)
    Yl = list(Imp.values())
    Y = np.array(Yl)
    if X.size!=0:
        inter, slope = LR(X, Y)
    else:
        inter=0
        slope = 0
    # INSERT LINEAR REGRESSION - possibly fill in all values from 1-etc.
    for index in yeardict.values():
        if not index in Imp.keys(): # if it is an empty value
            Imp[index] = (slope*index) + inter #y = mx+b
        if Imp[index] == 'None':
            print("true it equals None")
            Imp[index] = (slope*index) + inter # should check if it even has a slope, does LR fail when empty list?

    return Imp


def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 

def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 


def LR(x, y): 
    # observations 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print(b)
    intercept = b[0] 
    slope = b[1]

  
    # plotting regression line 
    #plot_regression_line(x, y, b)
    return intercept, slope




#LatLon = (-120.3, 37.81)
#lon= -120
#lat = 37
#LL = (-120, 37)


def RemoveExt(path):
    sp = os.path.splitext(path)
    newpath = sp[0]
    return newpath

def GetTail(path):
    headtail = os.path.split(path)
    return headtail[1]
