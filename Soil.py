import requests
import numpy as np


#'https://rest.soilgrids.org/soilgrids/v2.0/properties/query?lon=-72&lat=-9&property=bdod&property=cec&property=cfvo&property=clay&property=nitrogen&property=ocd&property=ocs&property=phh2o&property=sand&property=silt&property=soc&depth=0-5cm&depth=0-30cm&depth=5-15cm&depth=15-30cm&depth=30-60cm&depth=60-100cm&depth=100-200cm&value=Q0.05&value=Q0.5&value=Q0.95&value=mean&value='

allprop = 'property=bdod&property=cec&property=cfvo&property=clay&property=nitrogen&property=ocd&property=ocs&property=phh2o&property=sand&property=silt&property=soc'
alldepths = 'depth=0-5cm&depth=5-15cm&depth=15-30cm&depth=30-60cm&depth=60-100cm&depth=100-200cm'
allvalues = 'value=Q0.05&value=Q0.5&value=Q0.95&value=mean&value=uncertainty'

parameters = []
properties = allprop
depths = alldepths
values = allvalues
LonLat = 'lon=-121.553&lat=38.1252'
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
    'bdod': 0,
    'cec': 1,
    'cfvo': 2,
    'clay': 3,
    'nitrogen': 4,
    'ocd': 5,
    'ocs': 6,
    'phh2o': 7,
    'sand': 8,
    'silt': 9,
    'soc': 10


}
Depths = {
    5: 0, 
    15: 1, 
    30: 2, 
    60  : 3, 
    100: 4, 
    200: 5
}

URL = 'https://rest.soilgrids.org/soilgrids/v2.0/properties/query?{LonLat}&{properties}&{depths}&{values}'.format(LonLat=LonLat, properties=properties, depths=depths,values=values)

soilinfo = requests.get(URL)
sijson = soilinfo.json()

props = sijson['properties']
propslayers = props['layers']
#print(props)
#print(propslayers[0]) #will be for looped $ for each property
#print(Names[propslayers[0]['name']])
#print(propslayers[0]['unit_measure']['mapped_units'])
#print(propslayers[0]['depths'][0]) #will be for looped # for each depth level
#print(propslayers[0]['depths'][0])

#Since this will likely only be used for ML and no displayed values, just turn it into an np array

SoilStats = np.zeros((len(Names), len(Depths)))
SoilUncertainty = np.zeros(SoilStats.shape)

print(len(propslayers))
print(len(Names))
print(len(Depths))
#print(len(propslayers[0]['depths']))
row, col = SoilStats.shape

for i in range(len(propslayers)):
    
    currentprop = propslayers[i]
    PropIndex = NamesInd[currentprop['name']]
    k = len(currentprop['depths'])
    print(currentprop['name'])
    if k == col:
        for j in range(col):
            SoilStats[PropIndex, j] = currentprop['depths'][j]['values']['mean']
            SoilUncertainty[PropIndex,j] = currentprop['depths'][j]['values']['uncertainty']
    else:
        for k in currentprop['depths']:
            index = Depths[k['range']['bottom_depth']]
            SoilStats[PropIndex, index] = k['values']['mean']
            SoilUncertainty[PropIndex, index] = k['values']['uncertainty']
print(SoilStats)
print(SoilUncertainty)