import pickle
import os
from SentinelLoadstutzv2 import*
import numpy as np
import pandas as pd
from RasterAPIv01 import*
from BundleGeneratorv03 import*
import re

"PSEUDOCODE"

"Iterate through CSVFiles\CornYieldUSA.csv"
# use the current CropYieldEx.py function to iterate through,   ut instead just extract the first 400, if that are not already downloaded
#   if other (combined) in item: skip





#  Process into array and save for later extraction by cropyieldprediction OData function

#crop = 'Corn'
#Ypkl = 'CSVFiles\CornY.pkl' # all counties in dict w/ format (year, county, units): (yield)
#yfile = open(Ypkl, 'rb')
#ydict = pickle.load(yfile)
#yfile.close()
#[print(y, end = ', ') for y in ydict.items()]


def CountyCoordDict(pklpath = ' '):
    '''

    '''
    print("COUNTYCOORDICT CALLED")
    if pklpath  == ' ':
        mypkl = 'CSVFiles/CountyCoordinates.pkl'
    else:
        mypkl = pklpath
    #mycsv = 'CSVFiles\County_Latitude___Longitude_Points_For_Each_County_Static.csv'
    mycsv = 'CSVFiles/CountyLatLonPoints.csv'
    mydict = dict()
    with open(mycsv, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            
            lon = rows[13].replace('Â°', '')
            #print(lon," lon")
            lon = lon.replace('â', '')
            lon = lon.replace('€“', '')
            lon = lon.replace('+', '')
            lat = rows[12].replace('Â°', '')
            #print(lat," lat")
            lat = lat.replace('€“', '')
            lat = lat.replace('+', '')
            lat = lat.replace('â', '')
            lon = '-'+ lon
            
            tmpcounty = re.sub('[^A-Za-z]', '', rows[3].upper())
            
            print(lon, lat, end=',')
            mydict[tmpcounty] = (lon, lat)

            #mydict = {(rows[3]).upper(): (str(rows[13][1:len(rows[13])]).replace('Â°', '').replace('€“', ''), str(rows[12][1:len(rows[12])])).replace('Â°', '').replace('€“', '') for rows in reader}
    a_file = open('CSVFiles/CountyCoordinates.pkl', "wb")
    pickle.dump(mydict, a_file)
    a_file.close()
    return mypkl



def pklLoad(mypkl):
    tmpfile = open(mypkl, 'rb')
    tmpdict = pickle.load(tmpfile)
    tmpfile.close()
    return tmpdict


def genDataDirs(directory = None, imdirname = 'SentImages_Training'):
    "Creates all the necessary directories in order to organize training, raster, image, data appropriately using getTData. directory specifies the folder in which all the directories will be created. imdirname is the name of the main directory that will be created to store satellite imagery."
    crops = ['Wheat', 'Avocado', 'Peppers', 'SugarCane', 'Corn']
    
    
    for crop in crops:
        if directory == None:
            newpaths = [os.path.join(crop, '/data/train'), os.path.join(crop, '/data/test'), os.path.join(crop, '/data/validation')]
        else:
            newpaths = [os.path.join(directory, crop, '/data/train'), os.path.join(directory, crop, '/data/test'), os.path.join(directory, crop, '/data/validation')]
        for p in newpaths:
            if not os.path.exists(p):
                os.makedirs(p)
    extras = [os.path.join(imdirname, 'Imagery'), 'CSVFiles/'] 
    for e in extras:
        if not os.path.exists(e):
            os.makedirs(e)

    return True



# I should have a new extraction Odata function to not include weather/soil things

def getTData(crop, batch_size, year, impath='SentImages_Training\Imagery', csvpath = 'CSVFiles', initial = False, flush=False):
    '''
    Collects and saves necessary data in directory to be extracted into parameters for training crop yield prediction models.

    Parameters:
    crop(str): Crop name as a string w/ first letter capitalized.
    batch_size(int): The number of 
    year(int):
    impath(str): Path where satellite imagery jp2's will be saved
    initial(bool): If true, creates initial directories.
    flush(bool): Not an available feature currently.

    Returns:
    batches(list): A list of counties whose data was downloaded and is now available for training usage.
    '''

    "MAIN QUESTION: HOW DOES IT KNOW TO SAVE SATELLITE IAMGERY IN THE CORRECT PATH????"


    ". crop is crop name. If flush=True, will delete rasters and satellite images after extracting Odata. "
    if not os.path.isdir(impath):
        os.makedirs(impath)
    IMPATH = impath
    username = 'croblitos' 
    pword = 'LucklessMonkey$30'
    print("CP ONE")
    sl = Sentinel2Loaderv2('', 
                    username, pword,
                    apiUrl='https://scihub.copernicus.eu/apihub/', showProgressbars=True, dateToleranceDays=15, loglevel=logging.DEBUG, savepath=IMPATH)

    print("CP TWO")
    # scrape initial
    if initial == True: # should rarely happen
        gencountypkl(crop=crop)
        newpickle = CountyCoordDict()
        print(newpickle)
        passfailpkl(crop=crop, maindir = csvpath)

    print("CP THREE")

    "Corn Y pkl vs CornYieldUSA, cornypkl = saved yield to county. CornYieldUSA is all of them"

    "ADDRESS THE PROBLEM THAT A NEW PKL FILE IS NOT BEING CREATED EVEN WHEN INITIAL = TRUE"

    ypkl = os.path.join(csvpath, '{crop}Y.pkl'.format(crop=crop)) # all counties in dict w/ format (year, county, units): (yield)
    ydict = pklLoad(ypkl)
    yfin = pklLoad('{crop}Counties.pkl'.format(crop=crop))
    cc = pklLoad(csvpath + '/CountyCoordinates.pkl')
    print(cc)
    print(year, "this is the year")
    batches = []
    optimaldates = [date(year,7,1), date(year,6,10), date(year,5,15), date(year,4,25), date(year,4,1)]
    itdict = {'WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE':'WW', 'WHEAT, SPRING, DURUM - YIELD, MEASURED IN BU / ACRE': 'DW', 'WHEAT, SPRING, (EXCL DURUM) - YIELD, MEASURED IN BU / ACRE': 'SW', 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE': 'Co'}


    for ct in ydict.keys():
        print("new ct loop in ydict.keys()")
        if (not ct[0] in yfin) and (not 'OTHER' in ct[1]):
            if str(ydict[ct]).replace('.', '').isnumeric():
                tmpct = ct[1]
                pf = pklLoad(csvpath + '/{crop}datapassfail.pkl'.format(crop=crop))
                if tmpct in pf.keys():
                    print("ALREADY ATTEMPTED", tmpct, 'with result', pf[tmpct])
                else:
                    pf[tmpct] = False

                    print(tmpct, "this is tmpct")
                    tc = (float(cc[tmpct][0]), float(cc[tmpct][1]))
                    print(tc, "this is tc")
                    s = 0.005
                    poly = Polygon(((tc[0]-s, tc[1]-s), (tc[0]-s, tc[1]+s), (tc[0]+s, tc[1]+s), (tc[0]+s, tc[1]-s), (tc[0]-s, tc[1]-s)))

                    newdir = 'Input_{county}_{year}'.format(county=tmpct, year=str(year))
                    newpath = os.path.join(crop, 'data', 'train', newdir, str(year)+itdict[ct[2]]+crop+tmpct) 
                    if not os.path.exists(newpath):
                        os.makedirs(newpath) # Corn\data\train\Input1_Yakima_2019\2019CoCornKittitas
                 
                    sentpaths = os.path.join(impath, newdir)
                    if not os.path.exists(sentpaths):
                        os.mkdir(sentpaths)
                    imgpaths = os.path.join(sentpaths, 'Imagery')
                    if not os.path.exists(imgpaths):
                        os.mkdir(imgpaths) # SentImages_Training\Imagery\Input1_Yakima_2019\Imagery
                    rpaths = os.path.join(sentpaths, 'Rasters', crop)
                    if not os.path.exists(rpaths):
                        os.makedirs(rpaths) # SentImages_Training\Imagery\Input1_Yakima_2019\Rasters\Avocados
                    for op in optimaldates:
                        "ADD IN FUNCTION TO SPECIFY SAVE PATH FROM GETPRODUCTBANDTILES FUNCTION"
                        opdt = op.strftime("%Y-%m-%d")
                        dtstr = op.strftime("%Y_%m_%d")
                        print(dtstr, "dtstr")
                        print(opdt, "opdt")
                        satpath = os.path.join(imgpaths, newdir, dtstr)
                        if not os.path.exists(satpath):
                            os.makedirs(os.path.join(satpath))
                    


                        b4path = sl.getProductBandTiles(poly, 'B04', '10m', opdt, override_path=satpath) # add in function to specify save path from here! Unless it's automatic rn?
                        b8path = sl.getProductBandTiles(poly, 'B08', '10m', opdt, override_path=satpath)
                        b5path = sl.getProductBandTiles(poly, 'B05', '20m', opdt, override_path=satpath)
                    
                    #updates pass/fail value
                    pf[tmpct] = True
                    pickle.dump(pf, csvpath + '/{crop}datapassfail.pkl'.format(crop=crop))

                # get all rasters for each county included 
                
                    fdict = getFips(b4path)
                    print(len(fdict.items()), "this is length of fdict")
                    for fipk in fdict:
                        fipv = fdict[fipk] # fipk is county name # fipv is county fipcode
                        if (type(fipv)==str):
                            if(not '_None.' in fipv): #deals w/ invalid parts in/out of country)
                                cdlurl = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/ExtractCDLByValues?file=https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_{year}_{fips}.tif&values={crop}'.format(year=year, fips=fipv, crop=crop)
                                r = requests.get(cdlurl)#, allow_redirects=True)
                                s = str(r.content)
                                print(s)
                                surl = s.split('returnURL>')[1].split('</')[0]
                                tif = requests.get(surl)        
                                text = year + ct + crop + fipk + '.tif'

                                tpath = os.path.join(rpaths, text)
                                #figure out how to make with to save space
                                if not os.path.exists(tpath):
                                    open(tpath, 'wb').write(tif.content)
                                    batches.append(fipk)
                                    print('New tif written at : ', tpath)
                                else:
                                    print('File Path already exists for ', tpath)
                            else:
                                print('Error Found in Response: ', fipv, fipk)
                        "SAve yield to text file"
                        # Wheat\data\train\Input9_SouthDakota_2019\2019WWWheatGrant\YIELD.txt
                        f = open(newpath+'/YIELD.txt', 'w')
                        f.write(ydict[ct])
                        f.close()


                    "this is the major part i need to test!!!!"
                    BundleGenerate(newdir, datatype='train', Crop=crop) # indices saved
                    # save 




                    "OData from BGv03 to put yield, parameters into df"

                
                
                if len(batches) > batch_size:
                    # begin updating pkl
                    updatecountypkl()
                    return batches

                




    "This will still be completed in OTraining or whatever"
# For each county in those images, get pixels from bands/crop raster
# Process those images/yields into a npz
# add those counties to ydict
# Repeat loop for range(len(minibatch))





def GetYield(crop, pkl):
    '''
    Creates a YIELD.txt file in a corresponding folder to be used as ground truth for crop yield prediction.

    Parameters:
    crop(str): 
    pkl(str): Path to the yield pkl.

    Return:
    Void 
    '''

    print("getyield called")

    yfile = open(pkl, 'rb')
    ydict = pickle.load(yfile)
    yfile.close()
    print(len(ydict), "LENGTH OF YDICT")
    print(ydict)
    #iterate through rows in column data item 
    count = 0
    if crop == 'Wheat':
        p1 = 'Wheat/data/train'
        for fol in os.listdir('SentImages_Training\Imagery'):
            for tif in os.listdir(os.path.join('SentImages_Training\Imagery', fol, 'Rasters', crop)):
                tif = os.path.splitext(tif)[0]
                print(tif)
                print(tif[4:6])
                if tif[4:6] == 'SW':
                    cr = 'WHEAT, SPRING, DURUM - YIELD, MEASURED IN BU / ACRE'
                if tif[4:6] == 'WW':
                    cr = 'WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE'
                if tif[4:6] == 'DW':
                    cr = 'WHEAT, SPRING, DURUM - YIELD, MEASURED IN BU / ACRE'
                cnty = tif[11:] 
                key = (tif[0:4], cnty.upper(), cr)
                print(key)

                if key in ydict.keys():
                    print("TRUEKEY", key)
                    count = count+1
                    print(count, "COUNT")
                    y = ydict[key]
                    p2 = os.path.join(p1, fol, tif)
                    f = open(p2+'/YIELD.txt', 'w')
                    f.write(y)
                    f.close()
                    print(p2, "path")
                "APPEND TO PANDAS DATAFRAME AT THE END"

    if crop== 'Corn':
        p1 = 'Corn/data/train'
        for fol in os.listdir('SentImages_Training\Imagery'):
            for tif in os.listdir(os.path.join('SentImages_Training\Imagery', fol, 'Rasters', crop)):
                tif = os.path.splitext(tif)[0]
                cr = 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'
                cnty = tif[10:]
                key = (tif[0:4], cnty.upper(), cr)
                if key in ydict.keys():
                    print("TRUEKEY", key)
                    count = count+1
                    print(count, "COUNT")
                    y = ydict[key]
                    p2 = os.path.join(p1, fol, tif)
                    f = open(p2+'/YIELD.txt', 'w')
                    f.write(y)
                    f.close()
                    print(p2, "path")
                    "APPEND TO PANDAS DATAFRAME AT THE END"   
    print(p2, "path")
    #iterate through rasters, then put them in the appropriate crop folder
    yfile.close()  

def passfailpkl(crop, maindir=' '):
    '''
    If used, creates a passfail dictionary to be used to save previous attempts at downloading a county to avoid repeating counties w/out data. Structure is ['county'] = False once a download is attempted, and ['county'] = True once a download is successful.

    Parameters:
    crop(str): Crop name.
    maindir(str): Path to the directory where passfail dictionary will be saved

    Returns:
    Void
    '''

    pf = dict()
    pf['NULLCOUNTY'] = False
    if maindir == ' ':
        path = os.path.join(crop+'datapassfail.pkl')
    else:
        path = crop+'datapassfail.pkl'

    a_file = open(path, 'wb')
    pickle.dump(pf, a_file)
    a_file.close()
    


def gencountypkl(crop):
    "Used for this initial scrape of counties that have already been extracted appropriately"
    print("gencountypkl called")
    
    counties= dict()
    if crop.lower() == 'wheat':
        p1 = 'Wheat/data/train'
        path = 'WheatCounties.pkl'
    if crop.lower() == 'corn':
        p1 = 'Corn/data/train'
        path = 'CornCounties.pkl'
    for fol in os.listdir(p1):
        for cty in os.listdir(os.path.join(p1, fol)):
            county = cty.lower().split(crop.lower(), 1)[1]
            #county = county.upper()
            county = county[0].upper() + county[1:len(county)].lower()
            counties[county] = os.path.join(p1, fol)
    
    a_file = open(path, "wb")
    pickle.dump(counties, a_file)
    a_file.close()


"Pay attention to if these are in lower case!"
def updatecountypkl(pkl, newdict):
    print("updatecountypkl called")
    "Updates dictionary of extracted counties by county name and path"
    yfile = open(pkl, 'rb')    
    ydict = pickle.load(yfile)
    yfile.close()
    for k, v in newdict.items():
        ydict[k] = v
    afile = open(pkl, 'wb')
    pickle.dump(ydict, pkl)
    afile.close()

    

            # the string after "crop"
            # how to get the index of a string  

    



    
