from CTrainingHelpers import*
from BundleGeneratorv03 import*
from RasterAPIv01 import*
from indicesformulas import*
from SentinelLoadstutzv2 import*
from CropYieldPrediction import*
from CropStagePrediction import*



#def main():
 #   autodata(None)
    #trainmodels()




def autodata(crop, maindirectory, make_directories = False, initialscrape = False, batch_size=1):

    '''
    Collects all parameters necessary to build, and organizes them into the appropriate directory structure.

    Parameters:
    crop(str): 
    maindirectory(str): 
    make_directories(bool): If true, makes the directory structure where all training data and images will be stored. 
    initialscrape(bool): If true, a new dictionary containing county and yield information is saved as a pkl for usage. This should only be used if collecting from a new year or new crop(currently 2019 exists for corn and wheat)
    batch_size()


    
    Returns:


    '''

    "Scrape collects . Batches is the number of new counties to add to the training data. Year is the year for which data will be collected. Functionally, only 2019 is available. H"
    if initialscrape == True:
        initial = True
        
    else:
        initial = False
    if make_directories == True:
        genDataDirs(directory=maindirectory)


    getTData(crop=crop, batch_size=batch_size, year = 2019, initial=initial, flush=False)


def trainmodels(crop, YP = True, SP = True, presaved=False):
    '''
    Trains machine learning models using in standard directory created by autodata/genDataDirs.

    Parameters:
    crop(String): This is the crop written with a capitalized first letter.
    YP(bool): If true, trains yield prediction model.
    SP(bool): If true, trains crop stage prediction model.
    presaved(bool): If true, accesses a presaved dataframe containing training data from an earlier function call in order to reduce runtime. IF you have new data that has not been passed, specify as false.

    Returns: 
    Returns True
    '''
    if YP == True:
        CYPTrain(crop, presaved)
    if SP == True:
        CSPTrain(crop, presaved)
    return True


crop = 'Wheat'
autodata(crop, None, initialscrape=True)
trainmodels(crop)
