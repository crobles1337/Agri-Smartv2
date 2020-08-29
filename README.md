Last updated: 8/4/20
Rough*

Overview: 
Contains code to create the Agri Smart platform.




userQuery.py:
Code for an input query from a user. It simply requires a file with available coordinates, coordinates as a string in EPSG:4326, or a shape file, a tif/tiff file, or jp2 file. From there, satellite imagery is gathered, and . 
Currently has functionality to collect and output 
1. historical weather
2. a 7-day weather forecast
3. cold/heat stress events 
4. satellite imagery analyzed for NDVI, NDRE, CCCI, ARVI, SIPI, RECI, GCI, MI, NDWI, for current date. Some of these are stress indices (RECI, GCI, SIPI), others are moisture indices(MI, NDWI), and others are varied-usage vegetation indices (NDVI, CCCI, NDRE, ARVI)
5.  10 historical NDVI images, a historical productivity map created using an averaged overlay of the NDVI imagery
6. 5 historical NDRE images.
7. Soil info can be gathered and will be used as an input parameter for machine learning crop yield prediction, but soil info may be misleading to farmers as it is likely not highly reflective of farm soil being used.
8. Crop Stage prediction is pending (debugging helpers)
9. Crop Yield prediction is pending (need to significantly improve ML performance to be useful)

userQueryHelpers.py:
Contains helpers directly related to the user query, excluding helpers for machine learning processes. 




MLTest2:
Contains code that organizes all training data collected in SatelliteImages and respective crop training data. This will soon be deprecated and coded into Cropyield.py.

MLHelpers.py:
Currently incomplete. Contains helpers for userquery.py that help with ML functions, implementing and providing trained algoritmh for crop yield prediction and crop stage prediction.

CropStagePrediction.py:
CropStagePrediction contains current code to organize already collected training data, and input to train crop stage prediction training algorithm. Currently functional with available files. ML Algorithm to be tweaked, trained, and saved for use in MLHelpers.py to be used in Userquery.py



Bundle2Helpers.py:
Helpers to organize and creating training data inputs for Crop Yield prediction.








Training Data:
Training data was created with the following input parameters:
1. Historical weather (mintemp, maxtemp, precipitation, snow, snow depth, cold stress events, heat stress events)
2. Satellite crop indices (mean and quartile of collective crop values) including NDVI, NDRE, RECI, CCCI from 5 images from the past 4 months from when queried (usually summer)
3. Soil Stats collected from SoilInfo API (Deprecated)

Use the functions in CollectTraining.py, and see the google doc documentation for more details to building training data.
The training data generally came from NASS Cropland Data Layer in order to find exact crop coordinates.
Images were collected from Sentinel-2 satellite imagery using the SentinelSat Copernicus


The output ground truth label was gathered using NASS QuickStats which has labeled bu/acre crop yield by county in the USA. 
Training data is divided by specific crop. Currently there are 5 directories, however only 2 contain substantial training data, corn and wheat. Each crop directory contains test, train, and validation. train is currently the only non-empty folder. 
The 5 folders existing (all in .gitignore) are Wheat, Corn, Avocado, Peppers, and SugarCane.
The directory is organized into input batches, containing one satellite image (multiple bands) per folder in train, named as "year+2 letter crop label+ crop name + county". 
Inside of each of these folders are 3 csv's with temperature stress counts, historical weather with imputation, historical weather w/out imputation (raw), soilstats in an npz, and the associated uncertainty associated with those soil stats, and lastly there are 5 folders which contain satellite imagery (B04, B08, B5) at dates scattered from March/April to July/August labeled by there satellite imagery dating. 







