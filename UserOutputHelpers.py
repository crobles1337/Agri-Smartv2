import requests
import datetime
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
import pickle
import csv

# just prints week weather forecast
def GetWForecast(lat, lon, daily = True, hourly=False, minutely = False, current=False, tempunit = 'metric'):
    Days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    key = '35e02b4dab539973840fc771425f3539'

    if daily == True:
        WeatherStats = WeekForecast(lat, lon, key, tempunit)
        
        dayofweek = datetime.datetime.now().weekday()
        for i in range(7):
            print('The forecast for ', Days[(dayofweek+i)%7], 'is' )
            for k in WeatherStats:
                print(k, ': ', WeatherStats[k][i], WeatherStats[k][7]) 
    
        week = list()
        week.append('Weather Forecast')
        for i in range(7):
            dayofweek = Days[(datetime.datetime.now().weekday() + i)%7]
            week.append(dayofweek)
        week.append('Units')
        with open('forecast1.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([week])
            for k in WeatherStats:
                writer.writerow([k] + ["," + str(v) for v in WeatherStats[k]])
        


def WeekForecast(lat, lon, key, tempunit):
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

#FOR ANY POLYGON INPUT, JUST MAKE A LIST OF X, Y, AND THEN TAKE THE MIN, MAX OF EACH LIST GIVING YOU YOUR TOP, BOT, LEFT, RIGHT BOUNDS


def GetStationString(LatLon, MyAlt):
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
    xb = float(v[1])
    yb = float(v[0])
    dist =  ((LatLon[0] - xb)**2 + (LatLon[1]-yb)**2)**(1/2)
    #print(dist)
    return dist
# calculates difference in altitude (they are both in meters)
def AltDif(MyAlt, v):
    stationAlt = float(v[2])
    #check if either is missing value
    if ((MyAlt == -10000) or (stationAlt==-999)):
        return 0
    AltDif = abs(stationAlt - MyAlt)
    return AltDif

def GetHW(WS):
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
                #print(CelsiusTMAX, ": ", newWS[i]['DATE'])
            else:
                tmaxDict[newWS[i]['DATE']] = (CelsiusTMAX, 1)
                #print(CelsiusTMAX, newWS[i]['DATE'])

        if 'SNWD' in newWS[i]:
            if newWS[i]['DATE'] in snwdDict:
                SNWDCollisions = snwdDict[newWS[i]['DATE']][1] + 1
                snwdDict[newWS[i]['DATE']] = ((float(newWS[i]['SNOW'])*(1/SNWDCollisions)) + (snwdDict[newWS[i]['DATE']][0]*(1-(1/SNWDCollisions))), SNWDCollisions)
    return prcpDict, tmaxDict, tminDict, snowDict, snwdDict, 


def HeatStressCheck(croptype, tmax):
    
    HeatStressDict = {'corn': 35, 'avocado': 35, 'wheat': 32, 'sugarcane': 40, 'jalapeno': 32
    }
    stress = HeatStressDict[croptype]
    events = dict()
    for item in tmax:
        if tmax[item][0]>=stress:
            events[item] = tmax[item][0]
    count = len(events)
    return events, count
    

def ColdStressCheck(croptype, tmin):
    ColdStressDict = {'corn': 8, 'avocado': 0, 'wheat': 9, 'sugarcane': 18, 'jalapeno': 12 
    }
    stress = ColdStressDict[croptype]
    events = dict()
    for item in tmin:
        if tmin[item][0]<=stress:
            events[item] = tmin[item][0]
    count = len(events)
    return events, count

def RemoveCollisions(stats):
    # stats is list of dicts
    for d in stats:
        for v in d.keys():
            d[v] = d[v][0]
    return stats

def GetHistWeather(LatLon, start, end):
    LatLonString = str(LatLon[0]) + ','+ str(LatLon[1])
    MyAlt = GetAltitude(LatLon) #CHECK UNITS- meters for API
    
    stationstring = GetStationString(LatLon, MyAlt)
    dates = 'startDate={start}&endDate={end}'.format(start=start, end=end)
    WeatherStats = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations={stationstring}8&dataTypes=SNOW,PRCP,TMIN,TMAX,SNWD&{dates}&includeAttributes=true&includeStationName:1&includeStationLocation:1&format=json'.format(stationstring=stationstring, dates = dates))
    newWS = WeatherStats.json()
    
    prcpDict, tmaxDict, tminDict, snowDict, snwdDict = GetHW(newWS)

    cs, cscount, = ColdStressCheck('wheat', tminDict)
    hs, hscount = HeatStressCheck('wheat', tmax=tmaxDict)

    stats = [prcpDict, tmaxDict, tminDict, snowDict, snwdDict]
    stats = RemoveCollisions(stats)
    stats.append(cs)
    stats.append(hs)
    print(len(prcpDict))
    print(len(tminDict))
    print(len(tmaxDict))
    print(len(snowDict))
    print(len(snwdDict))

    #for stat in prcpDict:
        #this is me trying to write all stats together to the same CSV
    with open('stats1.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', ',Precipitation (mm)', ',Max Temp (C)', ',Min Temp (C)', ',Snowfall (mm)', ',Snow Depth(mm)', ',Cold Stress Events(C)', ',Heat Stress Events (C)'])
        for key in sorted(prcpDict.keys(), key=lambda x: x):
            writer.writerow([key] + ["," + str(d.get(key, None)) for d in stats])

def getImages(LatLon):
    exlon = LatLon[1]
    exlat = LatLon[0]
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
    band11 = rasterio.open(MyImagePath + "B11.jp2",  driver = 'JP2OpenJPEG') #swir1 - 20m
    band12 = rasterio.open(MyImagePath + "B12.jp2",  driver = 'JP2OpenJPEG') #swir2 - 20m
    bandTCI = rasterio.open(MyImagePath + "TCI.jp2",  driver = 'JP2OpenJPEG') 
    tci = bandTCI.read(1)
    pointbuffer = 700 #this is in 10 m units
    boxbuffer = 20 # this is in 10 m units
    #bbox = rasterio.warp.transform_bounds()

    exx, exy = rasterio.warp.transform(exinputCRS, band4.crs, [exlat], [exlon]) # check if lat, lon should be flipped

    row, col = band4.index(exx, exy)
    c = col[0]
    r = row[0]
    #print(r)
    #print(c)
    top = c+pointbuffer
    bot = c-pointbuffer
    left = r-pointbuffer
    right = r+pointbuffer
    #print(bot-top)
    #print(right-left)

    ogRedBand = band4.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    ogNIRBand = band8.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    rededge = band5.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
    ogGreenBand = band3.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    ogBlueBand = band2.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    wnnir = band8A.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
    wswir1 = band11.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
    wswir2 = band12.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')

    #Stretch all 20m bands to 10980x10980
    RedEdgeStretch = np.repeat(np.repeat(rededge,2, axis=0), 2, axis=1) 
    SWIR1Stretch = np.repeat(np.repeat(wswir1,2, axis=0), 2, axis=1) 
    SWIR2Stretch = np.repeat(np.repeat(wswir2,2, axis=0), 2, axis=1) 
    NNIRStretch = np.repeat(np.repeat(wnnir,2, axis=0), 2, axis=1) 

    ogRedEdgeStretch = RedEdgeStretch
    ogSWIR1Stretch = SWIR1Stretch
    ogSWIR2Stretch = SWIR2Stretch
    ogNNIRStretch= NNIRStretch

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

    band2.close()
    band3.close()
    band4.close()
    band5.close()
    band8.close()
    band8A.close()
    band11.close()
    band12.close()
    
    matplotlib.image.imsave('SampleImages/MapadeHumedadejemplo.png', nNDWI)
    matplotlib.image.imsave('SampleImages/NDVIejemplo.png', nNDVI)
    matplotlib.image.imsave('SampleImages/NDRE_etapatardia.png', nNDRE)
    matplotlib.image.imsave('SampleImages/SAVI_etapatemprana.png', nSAVI)
    matplotlib.image.imsave('SampleImages/RECI_estres.png', nRECI)
    matplotlib.image.imsave('SampleImages/ARVIejemplo.png', nARVI)
    matplotlib.image.imsave('SampleImages/GVMI1_humedad.png', nGVMI)
    matplotlib.image.imsave('SampleImages/NDMI_humedad.png', nNDMI)
    matplotlib.image.imsave('SampleImages/GCI_estres.png', nGCI)
    

    #savez_compressed('SampleImages/samplendvi1.npz', nNDVI)
    #savez_compressed('SampleImages/samplendre1.npz', nNDRE)
    #savez_compressed('SampleImages/samplesavi1.npz', nSAVI)
    #savez_compressed('SampleImages/samplesipi1.npz', nSIPI)
    #savez_compressed('SampleImages/samplearvi1.npz', nARVI)
    #savez_compressed('SampleImages/samplegvmi1.npz', nGVMI)
    #savez_compressed('SampleImages/samplendmi1.npz', nNDMI)
    #savez_compressed('SampleImages/samplegci1.npz', nGCI)
    #savez_compressed('SampleImages/sampleMoistureMap1.npz', nNDWI) #MOISTURE INDEX AND NDWI ARE SAME THING!




# just switch it to whatever coordinate system is used to query satellite imagery
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

    

start = '2020-01-01' # will prolly be this
end = '2020-07-07' # will prolly be to mid august
# end should equal current date! or datetime.now to date()
LatLon = (-120.3, 37.81)
lon= -120
lat = 37
LL = (-120, 37)



def getAll():
    LatLon = (-120.3, 37.81)
    lon= -120
    lat = 37
    WeatherCall(lat, lon)
    getImages(LatLon)

