import requests
import numpy as np
from numpy import save
from numpy import savetxt
from numpy import savez_compressed
from numpy import load
import os
from Bundle2Helpers import GetAltitude, GetHW, GetStationString, CalcDist, AltDif, HeatStressCheck, ColdStressCheck, GetSoil, GetHistWeather, SaveSoil

"Pseudocode"

#1. Index to the appripriate bundle 1 or iterate through bundle 1s
##NPZs/data/{datatype}/Input_{county}_{year} accesses maindirectory


#2. Soil Stats
# # input coordinates, extract these indices into probably a csv b/c its not much information.

#3. Historical Weather
# # Cold/Heat Stress Events 
# # Input coordinates, get weather data organized into a csv. Check the amount of space this requires. 
# # In a separate file, 2 csv entries, 1 is Cold Stress Events: value, and the other is Heat Stress Events: value


#4. Index the appropriate county-year-crop yield or find it directly using crop yield API 

def main():
    
    #ultimately, lonlat, latlon, and path should be determined by the other
    # this means, it might be best, easiest and most efficent to do all of these simultaneously with bundle1generation
    path = 'Bundle2Test'
    LonLat = 'lon=-121.553&lat=38.1252'
    LatLon = (38.12, -121.55)
    start = '2019-01-01' # will prolly be this
    end = '2019-08-01' # will prolly be to mid august
    GetSoil(LonLat, path) # this should save the files, maybe I should return a filepath?
    GetHistWeather(LatLon, start, end, path) # this should save as well, maybe return filepath?
    





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
    stats = [prcpDict, tmaxDict, tminDict, snowDict, snwdDict]
    cs, cscount, = ColdStressCheck('wheat', tminDict)
    hs, hscount = HeatStressCheck('wheat', tmax=tmaxDict)
    

    
    #for stat in prcpDict:
        #this is me trying to write all stats together to the same CSV
    
    hw = 'HistWeather'
    hwpath = os.path.join(path, hw)
    sc = 'Stress_Count'
    scpath = os.path.join(path, sc)

    with open(hwpath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', ',Precipitation (mm)', ',Max Temp (C)', ',Min Temp (C)', ',Snowfall (mm)', ',Snow Depth(mm)'])
        for key in sorted(prcpDict.keys(), key=lambda x: x):
            writer.writerow([key] + ["," + str(d.get(key, None)) for d in stats])
    

    with open(cspath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Cold Stress', 'Heat Stress'])
        writer.writerow(cs, hs)
#    for stat in stats:
#        #this is me trying to write all stats together to the same CSV
#        with open('directory/stats', 'w') as f:
#            for key in stat.keys():
#                f.write("%s,%s\n"%(key,stat[key]))

    



if __name__ == "__main__":
    main()