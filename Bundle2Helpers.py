import csv
import os
import pickle
import requests
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
from sklearn.linear_model import LinearRegression
from numpy import genfromtxt

import matplotlib.pyplot as plt    

#helpers for historical weather

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
    xb = float(v[0])
    yb = float(v[1])
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

def GetHW(WS, collisions=False):
    # Dicts to save dated values
    newWS = WS
    prcpDict = dict()
    tminDict = dict()
    snowDict = dict()
    tmaxDict = dict()
    snwdDict = dict()

    

    for i in range(len(newWS)):
        if 'PRCP' in newWS[i]:
            #if date already stored, average over values
            if newWS[i]['DATE'] in prcpDict:
                Collisions = prcpDict[newWS[i]['DATE']][1] + 1
                # date's value = new value* 1/collisions + old value* (1 - 1/collisions) 
                prcpDict[newWS[i]['DATE']] = ((float(newWS[i]['PRCP'])*(1/Collisions)) + (prcpDict[newWS[i]['DATE']][0]*(1-(1/Collisions))),    Collisions)
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
    # removes collisions from values
    if collisions==False:
        for k in snwdDict.keys():
            if type(snwdDict[k])==tuple:
                snwdDict[k] = snwdDict[k][0]
        for k in tminDict.keys():
            if type(tminDict[k])==tuple:
                tminDict[k] = tminDict[k][0]
        for k in tmaxDict.keys():
            if type(tmaxDict[k])==tuple:
                tmaxDict[k] = tmaxDict[k][0]
        for k in prcpDict.keys():
            if type(prcpDict[k])==tuple:
                prcpDict[k] = prcpDict[k][0]
        for k in snowDict.keys():
            if type(snowDict[k])==tuple:
                snowDict[k] = snowDict[k][0]

    
    
    return prcpDict, tmaxDict, tminDict, snowDict, snwdDict, 


def HeatStressCheck(croptype, tmax):
    
    HeatStressDict = {'corn': 35, 'avocado': 35, 'wheat': 32, 'sugarcane': 40, 'jalapeno': 32
    }
    stress = HeatStressDict[croptype]
    events = dict()
    for item in tmax:
        if tmax[item]>=stress:
            events[item] = tmax[item] # eliminated [0] cuz no collisions
    count = len(events)
    return events, count
    

def ColdStressCheck(croptype, tmin):
    ColdStressDict = {'corn': 8, 'avocado': 0, 'wheat': 9, 'sugarcane': 18, 'jalapeno': 12 
    }
    stress = ColdStressDict[croptype]
    events = dict()
    for item in tmin:
        if tmin[item]<=stress:
            events[item] = tmin[item] # eliminated [0] cuz no collisions
    count = len(events)
    return events, count


def GetSoil(LonLat, path):
    allprop='property=bdod&property=cec&property=cfvo&property=clay&property=nitrogen&property=ocd&property=ocs&property=phh2o&property=sand&property=silt&property=soc'
    alldepths = 'depth=0-5cm&depth=5-15cm&depth=15-30cm&depth=30-60cm&depth=60-100cm&depth=100-200cm'
    allvalues = 'value=Q0.05&value=Q0.5&value=Q0.95&value=mean&value=uncertainty'
    Names = {
    'bdod': 'Bulk density of the fine earth fraction',
    'cec': 'Cation exchange capacity of the soil',
    'cfvo': 'Volumetric fraction of coarse fragments',
    'clay': 'Proportion of clay particles in the fine earth fraction',
    'nitrogen': 'Total nitrogen',
    'phh2o': 'Soil pH',
    'sand': 'Proportion of sand particles in the fine earth fraction',
    'silt': 'Proportion of silt particles in the fine earth fraction',
    'soc': 'Soil organic carbon in the fine earth fraction',
    'ocd': 'Organic carbon density',
    'ocs': 'Organic carbon stocks'
    }
    NamesInd = {
        'bdod': 0, 'cec': 1, 'cfvo': 2,
        'clay': 3, 'nitrogen': 4, 'ocd': 5,
        'ocs': 6, 'phh2o': 7, 'sand': 8,
        'silt': 9, 'soc': 10
    }
    Depths = {
        5: 0, 15: 1, 30: 2, 60 : 3, 100: 4, 200: 5
    }
    parameters = []
    properties = allprop
    depths = alldepths
    values = allvalues
    LonLat = LonLat
    URL = 'https://rest.soilgrids.org/soilgrids/v2.0/properties/query?{LonLat}&{properties}&{depths}&{values}'.format(LonLat=LonLat, properties=properties, depths=depths,values=values)
    soilinfo = requests.get(URL)
    sijson = soilinfo.json()
    print(sijson, LonLat, path)
    props = sijson['properties']
    propslayers = props['layers']
    SoilStats = np.zeros((len(Names), len(Depths)))
    SoilUncertainty = np.zeros(SoilStats.shape)
    row, col = SoilStats.shape

    for i in range(len(propslayers)):    
        currentprop = propslayers[i]
        PropIndex = NamesInd[currentprop['name']]
        k = len(currentprop['depths'])
        if k == col:
            for j in range(col):
                SoilStats[PropIndex, j] = currentprop['depths'][j]['values']['mean']
                SoilUncertainty[PropIndex,j] = currentprop['depths'][j]['values']['uncertainty']
        else:
            for k in currentprop['depths']:
                index = Depths[k['range']['bottom_depth']]
                SoilStats[PropIndex, index] = k['values']['mean']
                SoilUncertainty[PropIndex, index] = k['values']['uncertainty']
    S = 'SoilStats'
    U = 'SoilUncertainty'
    
    pathS = os.path.join(path, S)
    pathU = os.path.join(path, U)
    SaveSoil(data=SoilStats, path = pathS)
    SaveSoil(data=SoilUncertainty, path=pathU)

def SaveSoil(data, path):
    savez_compressed(path, data)

def GetHistWeather(LatLon, start, end, path):
    LatLonString = str(LatLon[0]) + ','+ str(LatLon[1])
    MyAlt = GetAltitude(LatLon) #CHECK UNITS- meters for API
    
    stationstring = GetStationString(LatLon, MyAlt)
    dates = 'startDate={start}&endDate={end}'.format(start=start, end=end)
    WeatherStats = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations={stationstring}8&dataTypes=SNOW,PRCP,TMIN,TMAX,SNWD&{dates}&includeAttributes=true&includeStationName:1&includeStationLocation:1&format=json'.format(stationstring=stationstring, dates = dates))
    newWS = WeatherStats.json()
    
    prcpDict, tmaxDict, tminDict, snowDict, snwdDict = GetHW(newWS)
    statsraw = [prcpDict, tmaxDict, tminDict, snowDict, snwdDict]
    cs, cscount, = ColdStressCheck('wheat', tminDict)
    hs, hscount = HeatStressCheck('wheat', tmax=tmaxDict)

  ####################################################################  
    if '2019' in end:
        year = '2019'
    if '2018' in end:
        year='2018'
    if '2020' in end:
        year='2020'

    yeardict = GetYearDict(year)
    ImpPrcp = dict()
    ImpTmin = dict()
    ImpTmax = dict()
    ImpSnow = dict()
    ImpSnwd = dict()
    ImpPrcp = GetLR(yeardict, prcpDict, ImpPrcp)
    ImpTmin = GetLR(yeardict, tminDict, ImpTmin)
    ImpTmax = GetLR(yeardict, tmaxDict, ImpTmax)
    ImpSnow = GetLR(yeardict, snowDict, ImpSnow)
    ImpSnwd = GetLR(yeardict, snwdDict, ImpSnwd)
    stats = [ImpPrcp, ImpTmax, ImpTmin, ImpSnow, ImpSnwd]

    hwr = 'HistWeatherRaw.csv'
    hwrpath = os.path.join(path, hwr)
    hw = 'HistWeather.csv'
    hwpath = os.path.join(path, hw)

    sc = 'Stress_Count.csv'
    scpath = os.path.join(path, sc)

    # Write the raw, possibly incomplete values
    with open(hwrpath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', ',Precipitation (mm)', ',Max Temp (C)', ',Min Temp (C)', ',Snowfall (mm)', ',Snow Depth(mm)'])
        for key in sorted(prcpDict.keys(), key=lambda x: x):
            writer.writerow([key] + ["," + str(d.get(key, None)) for d in statsraw])
    #Write weather to csv including imputation values
    with open(hwpath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', ',Precipitation (mm)', ',Max Temp (C)', ',Min Temp (C)', ',Snowfall (mm)', ',Snow Depth(mm)'])
        for key in sorted(ImpPrcp.keys(), key=lambda x: x):
            writer.writerow([key] + ["," + str(d.get(key, None)) for d in stats])


    with open(scpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Cold Stress', 'Heat Stress'])
        writer.writerow([cscount, hscount])
#helpers for...




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


def GetYearDict(year):
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


def GetLR(yeardict, statDict, Imp):
    "Uses linear regression to perform imputation on missing values. Returns new list of values."

    for key in yeardict.keys():
        if key in statDict:
            Imp[yeardict[key]] = statDict[key]

    Xl = list(Imp.keys())
#    print("Xl", Xl)
    X = np.array(Xl)
    Yl = list(Imp.values())
    Y = np.array(Yl)
#    print("x", X)
    if X.size!=0:
        inter, slope = LR(X, Y)
    #contingency for completely empty values
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