import os
import numpy as np
import pandas as pd





"Work on this once you have played with the ML more"
def inputtoCS():
    "Prepares input to Crop Stage Prediction"
    columns = 8
    count = 0
    nr = load(ndrepath)
    ndre = nr['arr_0']
    nv = load(ndvipath)
    ndvi = nv['arr_0']

    tlist = []
    if ndre.size!=0:
        ndres = [np.average(ndre), np.quantile(ndre, .25), np.quantile(ndre, .5), np.quantile(ndre, .75)]
        tlist.extend(ndres)
    else:
        raise Exception('Error: NDRE is empty')
    if ndvi.size!=0:
        ndvis = [np.average(ndvi, 1), np.quantile(ndvi, .25), np.quantile(ndvi, .5), np.quantile(ndvi, .75)]
        tlist.extend(ndvis)
    else:
        raise Exception('Error: NDVI is empty')
    df = pd.DataFrame(columns=range(columns))
    df = loc[0] = tlist

    return df

def predictCS(input, crop = "Wheat"):
    "Applies trained algorithm to predict crop stage for specific crop. "

