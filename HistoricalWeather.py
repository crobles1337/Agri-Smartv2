"Main Resources"
"https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation"
"https://www.ncei.noaa.gov/support/access-search-service-api-user-documentation"
"https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf"
"ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/"
"https://www.climate.gov/maps-data/dataset/daily-temperature-and-precipitation-reports-data-tables"
"https://www.ncdc.noaa.gov/cdo-web/datasets"
import requests
import numpy as np
import pickle


#Current plan:
"Find all stations within a 5 euclidian coordinate units away from coordinate specified from text files"
"Take all, and average any dates where there is overlap."
"Give each dataset a fullness score, indicating how accurate the data might be because some is rather empty"
"Average, and feed those into weekly/monthly averages to reduce resolution."
"Use this data to identify any extreme heat/cold events"

LatLon = (37, -120)
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

#CHECK UNITS- meters for API

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
    if ((Altitude == -10000) or (stationAlt==-999)):
        return 0
    AltDif = abs(stationAlt - Altitude)
    return AltDif

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

dates = 'startDate=2020-01-01&endDate=2020-07-07'

WeatherStats = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations={stationstring}8&dataTypes=SNOW,PRCP,TMIN,TMAX,SNWD&{dates}&includeAttributes=true&includeStationName:1&includeStationLocation:1&format=json'.format(stationstring=stationstring, dates = dates))

newWS = WeatherStats.json()

# Dicts to save dated values
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
            prcpDict[newWS[i]['DATE']] = ((float(newWS[i]['PRCP'])*(1/Collisions)) + (prcpDict[newWS[i]['DATE']][0]*(1-(1/Collisions))), Collisions)
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
print(prcpDict)

PRCPSort = sorted(prcpDict)
TMINSort = sorted(tminDict)
TMAXSort = sorted(tmaxDict)
SNOWSort = sorted(snowDict)
SNWDSort = sorted(snwdDict)
print(type(PRCPSort[1]), "typehereprcpsortlist") # this is a string. Meaning sorted gives list of strings(only the keys), could be used to index dict in a sorted order

print(TMINSort)

HeatStressDict = {'CornHS': 35, 'AvocadoHS': 35, 'WheatHS': 32, 'SugarCaneHS': 40, 'JalapenoHS': 32
}
ColdStressDict = {'CornCS': 10, 'AvocadoCS': 0, 'WheatCS': 12, 'SugarCaneCS': 21, 'JalapenoCS': 15 
}
#HSDict = {35: 'CornHS', 35: 'AvocadoHS': 35, 'WheatHS': 32, 'SugarCaneHS': 40, 'JalapenoHS': 32
#}
#CSDict = {10:'CornCS', 0: 'AvocadoCS', 12: 'WheatCS', 21: 'SugarCaneCS', 15: 'JalapenoCS' 
#}

CornHS = 35 
CornCS = 10
AvocadoHS = 32
AvocadoCS = 0
WheatHS = 32
WheatCS = 12
SugarCaneHS = 40 
SugarCaneCS = 21
JalapenoHS = 32
JalapenoCS = 15

JalapenoStress = (0, 0)
SugarCaneStress = (0,0)
WheatStress = (0,0)
AvocadoStress = (0,0)
CornStress = (0,0)

print("PRCP, mm")
#for element in prcpDict:
#    print(element, ":", prcpDict[element])
#print("TMIN, celsius")
#for element in tminDict:
#    for k in ColdStressDict:
#        if tminDict[element][0]<ColdStressDict[k]:
#            print(k)
#    print(element, ":", tminDict[element])

#print("TMAX, celsius")
#for element in tmaxDict:
#    for k in HeatStressDict:
#        if tmaxDict[element][0]>HeatStressDict[k]:
#            print(k)
#    print(element, ":", tmaxDict[element])
#print("SNOW")
#for element in snowDict:
#    print(element, ":", snowDict[element]) 
#print("SNWD, mm")
#for element in snwdDict:
#    print(element, ":", snwdDict[element])






"hOW DO I SORT DICTIONARY?"
"HOW DO I ENSURE INPUT IS CHRONOLOGICALLY ORGANIZED, AND THAT THAT MATTERS"