import os
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from numpy import save
from numpy import savez_compressed
from numpy import load
import csv
import matplotlib
from Bundle2Helpers import *
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from numpy import load
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import math
from userQueryhelpers import*
	




####### Crop Yield Functions ############

def getcyinput(cypaths):
	'''
	Collects 80 parameters for yield prediction model from the specified paths.
	
	Parameters:
	cypaths(list): List of strings of 20 paths to 4 spectral indices from 5 timepoints.

	Returns:
	cyinput(df): A dataframe of the 80 parameters for the yield prediction model.

	'''
	cyi = []
	for path in cypaths:
		tempload = load(path)
		cyi.extend(tempload['arr_0'])
	print("below is cy input")
	[print(yi, end=',') for yi in cyi]
	print("above is cy input")
	cyinput = pd.DataFrame([cyi])
	return cyinput

    

def predictyield(data, crop):
	'''
	Uses trained algorithm to predict crop yield -- optimized for predictions during late June early July.

	Parameters:
	data(df): A pandas dataframe object of length 80
	crop(str): A string of the crop for which the yield prediction will be applied with the first letter capitalized. (ie. 'Corn')

	Returns:
	yi(float): A rounded, formatted float of the predicted yield.

	'''


	
	if crop.lower() == 'corn':
		mod = keras.models.load_model('models\corn\cropyieldcorn.HDF5')
		
	elif crop.lower() == 'wheat':
		mod = keras.models.load_model('models\wheat\cropyieldWheatv1.HDF5')

	yi = mod.predict(data)
	print(yi, "this is yi")
	yi = yiformat(yi, crop)
	
	cf = loaddf('{crop}TData.npz'.format(crop=crop))
	end = 80
	start = 0
	XX = cf.iloc[:, start:end]

	controlyi = mod.predict(XX)
	print(controlyi, "this is controlyi")

	return yi


def yiformat(yi, crop):
	'''
	Formats predicted bu/acre yield.
	
	Parameters:
	yi(list): A pre-formatting list containing predicted yield.
	crop(str): A string of the crop for which the yield prediction will be applied with the first letter capitalized. (ie. 'Corn')

	Returns:
	yi(float): A rounded, formatted float of the predicted yield.

	'''
	if crop.lower() == 'corn':
		y = round(yi[0][0])
		y = abs(y)
		return y
	else:
		return 0




######## Crop Stage Functions ###########
def getcstinput(cspaths):
	'''
	Collects and formats parameters as inputs to crop stage predictions

	Parameters:
	cspaths(list): A list of paths as strings to 2 npz's NDRE, and NDVI.

	Returns:
	csinput(array): 
	'''
	cs = []
	for path in cspaths:
		tempload = load(path)
		cs.extend(tempload['arr_0'])
	csinput = np.array(cs)
	return csinput

	


def predictstage(data, crop):
	if (crop.lower()) == 'corn':
		mod = keras.models.load_model('models\corn\cropstagecorn.HDF5')
	else:
		return 0
	print(data.shape)
	print(len(data))
	datadf = pd.DataFrame([data])
	print(datadf)
	print(datadf.shape)
	st = mod.predict(datadf)
	stage = stformat(st, crop)

	return stage

def stformat(st, crop):
	"Includes all processing to change raw model crop stage prediction to human-usable format."
	stage = math.ceil(st[0])
	return stage




############ other ##################

def dtfrompath(path):
	'''
	Extracts date from the specific path structure used to save jp2 images downloaded from Copernicus.
	
	Parameters:
	path: Path to a satellite image saved as a jp2.

	Returns:
	idate: A date type object with the date of the satellite image from the specified path.

	'''
	d = os.path.split(path)
	dat = d[1][0:10]
	dparts = dat.split('-')
	for i in range(len(dparts)):
		if int(dparts[i][0])==0:
			dparts[i] = dparts[i][1]

	idate = date(int(dparts[0]), int(dparts[1]), int(dparts[2]))
	return idate

def genstats(arr):
	"returns average, and 3 quantiles to be used for inputs"
	stats = [np.average(arr), np.median(arr), np.quantile(arr, .25), np.quantile(arr, .75)]
	return stats

def savedf(df, crop):
	filename = '{crop}TData.npz'.format(crop=crop)
	df.to_pickle(filename)
	return filename
def loaddf(filename):
	cf = pd.read_pickle(filename)
	return cf	


"Save band paths in a dict of dicts w/in userdirectory (first layer is recency(deltas in the past)) and then index. Index this dict of dicts and you'll get the band path to NDVI from current date. dd[0]['ndvi'] = userdir/band/path/tondvi"