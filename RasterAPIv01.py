import requests
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.warp
import os


#cdlurl = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/ExtractCDLByValues?file=https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/#CDL_{year}_{fips}.tif&values={crop}'.format(year=year, fips=fips, crop=crop['WW'])
#request2 = requests.get(cdlurl, allow_redirects=True)
#print(request2.content)
#mystr = str(request2.content)
#print(mystr.split('returnURL>')[1].split('</')[0])
#rfurl = mystr.split('returnURL>')[1].split('</')[0]
#rf = requests.get(rfurl)
#open('rf.tif', 'wb').write(rf.content)







#  CURRENTLY ONLY FOR WHEAT
def Rastertif(ctype, ct, ipre, iext):
    print("rastertif called!")
    cropdic = {'WW': '24', 'DW': '22', 'Co': '1', 'Pe':'216', 'Av':'215', 'SC':'45', 'SW':'23'}

    crop = cropdic[ct]  
    # getting ipath should be automated?
   # ipre = 'SentImages_Training\Imagery\Input1_Yakima_2019'
   # iext = 'Imagery\Yakima_2019_04_08\T10TFS_20190408T185919_B04_10m.jp2'
   
    ipath = os.path.join(ipre, iext)    
    rpath = MakeRasDir(ipre, ctype)
    if '2019' in ipath:
        year = '2019'
    if '2018' in ipath:
        year = '2018'
    print('cp1')
    fdict = getFips(ipath)
    print('cp2')
    for fipk in fdict:
        
        fipv = fdict[fipk] # fipk is county name # fipv is county fipcode
        print(fipk, fipv)
        if (type(fipv)==str):
            if(not '_None.' in fipv): #deals w/ invalid parts in/out of country

#   i can move nonecheck up here!
                print(year, fipv, crop)
                cdlurl = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/ExtractCDLByValues?file=https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_{year}_{fips}.tif&values={crop}'.format(year=year, fips=fipv, crop=crop)
                r = requests.get(cdlurl)#, allow_redirects=True)
                s = str(r.content)
                print(s)
                surl = s.split('returnURL>')[1].split('</')[0]
                tif = requests.get(surl)        
                text = year + ct + ctype + fipk + '.tif'

                tpath = os.path.join(rpath, text)
            #figure out how to make with to save space
                if not os.path.exists(tpath):
                    open(tpath, 'wb').write(tif.content)
                    print('New tif written at : ', tpath)
                else:
                    print('File Path already exists for ', tpath)
            else:
                print('Error Found in Response: ', fipv, fipk)

def MakeRasDir(ipath, ctype):
    print("makerasdir called")
    rpath = os.path.join(ipath, 'Rasters', ctype)
    if not os.path.exists(rpath):
        os.mkdir(rpath)
    return rpath


# Use https://geo.fcc.gov/api/census/#!/block/get_block_find API
def GetCounty(lat, lon):
    print("getcountycalled")
    #lat = '37'
    #lon = '-120'
    urlC = 'https://geo.fcc.gov/api/census/block/find?latitude={lon}&longitude={lat}&showall=true&format=json'.format(lat=lat, lon=lon)
    test = requests.get(urlC)
    cfip = test.json()['County']['FIPS']
    cname = test.json()['County']['name']
    return cfip, cname
    # this code 

def getFips(ipath):
    print("getfipscalled")
    #automate finding this path
    with rasterio.open(ipath) as b:
        bounds = b.bounds
        nbounds = rasterio.warp.transform_bounds(b.crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)
        left = nbounds[0] #lon
        bottom = nbounds[1] # lat
        right = nbounds[2] #lon
        top = nbounds[3] #lat
    fdict = dict()
    for i in range(8):
        lat = left + (right-left)/8*i
        for j in range(8):
            lon = bottom + (top-bottom)/8*j           
            fip, name = GetCounty(lat, lon)
            fdict[name] = fip

    return fdict



#Rastertif('Wheat', 'WW')

def GetRasterBatch(ctype, ct):
    pre = 'SentImages_Training\Imagery'
    skip = ['Input1_Yakima_2019', 'Input2_Sandusky_2019', 'Input3_LaCrosse_2019', 'Input4_Fresno_2019', 'Input5_Lubbock_2019', 'Input6_WPalmBeach_2019', 'Input7_BatonRouge_2019', 'Input8_Rapides_2019'] # Input1_Yakima_2019 is an example
     
    for dirp in os.listdir(pre):
        if 'Input' in dirp:
            if not dirp in skip:

                ipre = os.path.join(pre, dirp)
                ip1 = os.path.join(ipre, 'Imagery')
                iext = os.path.join('Imagery', os.listdir(ip1)[0]) #saves only ext
                ip2 = os.path.join(ip1, os.listdir(ip1)[0])
                for b in os.listdir(ip2):
                    if 'B04' in b:
                        iext = os.path.join(iext, b)

                Rastertif(ctype, ct, ipre, iext)

#GetRasterBatch('Wheat', 'SW')

#'b<faultstring>D:/CDL/NASS_DATA_CACHE/byfips/CDL_2019_None.tifdoes not exist</faultstring>'