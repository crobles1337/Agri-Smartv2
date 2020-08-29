import numpy as np
import requests
import csv
import pandas as pd
import pickle
import os


def MakeYDict(crop):
    if crop == 'Wheat':
        mycsv = 'CSVFiles\WheatYieldUSA.csv'
        mypkl = 'CSVFiles/WheatY.pkl'
    if crop == 'Corn':
        mycsv = 'CSVFiles\CornYieldUSA.csv'
        mypkl = 'CSVFiles/CornY.pkl'
    with open(mycsv, mode='r') as infile:
        reader = csv.reader(infile)
        #with open('CSVFiles\WeatherStations.csv', mode='w') as outfile:
        #writer = csv.writer(outfile)
        mydict = {(rows[1], rows[9], rows[16]): rows[19] for rows in reader}
    
    a_file = open(mypkl, "wb")
    pickle.dump(mydict, a_file)
    a_file.close()
    return mypkl


def GetYield(crop, pkl):
    yfile = open(pkl, 'rb')
    ydict = pickle.load(yfile)
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


crop = 'Corn'

pklname = MakeYDict(crop)


GetYield(crop, pklname)


