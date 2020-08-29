
# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import load
from numpy import genfromtxt
import pandas as pd
import os
from matplotlib import pyplot


# FOR CORN

"NEXT TASKS"

"KERAS WORKS VIA https://stackoverflow.com/questions/62465620/error-keras-requires-tensorflow-2-2-or-higher"




# load dataset
#dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
def LoadYData(crop):
	'''
	Loads Yield Data from the standardized directory structure into parameters for crop yield prediction model.

	Parameter:
	crop(str): First-letter capitalized string of crop name (ie. 'Corn')

	Return:
	
	'''
	print("LoadYData called", crop)
	c = 0
	emptycount =0
	# crop = 'Corn'
	mpath = os.path.join(crop, 'data/train')
	tarlist = list()
	hwlist = list()
	sclist = list()
	sslist = list()
	sulist = list()
	npilist = list()

	for fil in os.listdir(mpath): # fil = input
		p1 = os.path.join(mpath, fil)
		for codir in os.listdir(p1): # codir = 2019co
			if not 'YIELD' in codir:
				print("not yield in ", codir)
				p2 = os.path.join(p1, codir)
				tarray = [] # there is one tarray per county
				if 'YIELD.txt' in os.listdir(p2): # check if dataset is labeled
					print("yield.txt labeled in ", p2)
					with open(os.path.join(p2, 'YIELD.txt'), 'r') as y:
						yval = y.read()
					empty = False
					for date in os.listdir(p2): # date  = dated image
						if date != 'YIELD.txt':
							p3 = os.path.join(p2, date)
							tarpass = True
							if os.path.isdir(p3):
								for npz in os.listdir(p3):			
									print(os.path.splitext(npz)[0], "npz1")
									if os.path.splitext(npz)[1] == '.npz':
								
										ld = load(os.path.join(p3, npz))
										npi = ld['arr_0']
										print(npi.shape, "shape of ", p3, npz)
										if npi.shape[1] != 0:
											tarray.extend([np.average(npi), np.median(npi), np.quantile(npi, .25), np.quantile(npi, .75)]) # should end at size 4*16*5 = 320
										
											print("succes npz loading")
										else:
											print("shape is 0 for", p3, npz, print(npi.shape))
											empty = True
							else:
								tarpass = False
					npilist.append(len(tarray))
					if empty == False:
						wv = True
						# for each county we will also append
						# GET WEATHER HERE!!!
						p2 = str.replace(p2, 'Corn', 'Wheat', 1)
						p2 = str.replace(p2, 'CoCorn', 'WWWheat')
					
						print(p2)
						if not os.path.exists(p2): 
							print(p2, "doesn't exist")
							p2 = str.replace(p2, 'WWW', 'SWW')
						if not os.path.exists(p2):
							print(p2, "doesn't exist")
							p2 = str.replace(p2, 'SWW', 'DWW')
						if os.path.exists(p2):	
							listdir = os.listdir(p2)				
							print(p2, "this is the new p2")

					
							print(len(tarray), "length of tarray")
							#print(tarray.shape)
							"currently the tarray is empty??"
				
							#print(tarray.shape, "shape of tarray")
							tarray.append(float(yval))
						else: 
							print("CANNOT FIND WEATHERSTATS")
							wv = False

						if wv == True:
							if tarpass == True:
								if c == 0:
									cf = pd.DataFrame(columns = range(len(tarray)))
									cf.loc[c] = tarray
									c = c + 1
									dim = len(tarray)
								else:
									c = c+1
									cf.loc[c] = tarray
					else:
						print("error due to a value being empty")
						emptycount = emptycount+1
				else:
					print(p2, "yield text file doesn't exist and will not be added to the training set. Current length of training set == ", c)
				tarlist.append(len(tarray))
	print("tarlist", tarlist)
	print("count of 80 in tarlist", np.count_nonzero(tarlist==80))
	print("hwlist", hwlist)
	print("sclist", sclist)
	print("sulist", sulist)
	print("sslist", sslist)
	print("npilist", npilist)
	print(len(npilist))
	print(emptycount)
	print(c)
	print(dim)
	print("cf", cf, )#######################################
	savedf(df=cf, crop = crop)
	return cf, dim
# there are some errors b/c there are a lot of empty values rn
#I SHOULD MAKE A NEW FILE THAT DOES SAVE CF, B/C THIS TAKES LIKE 8 MIN TO MAKE EVERY TIME

def savedf(df, crop):
	print("savedf called")
	filename = '{crop}TData.npz'.format(crop=crop)
	df.to_pickle(filename)
	return filename
def loaddf(filename):
	cf = pd.read_pickle(filename)
	return cf	

def CYPTrain(crop, cfexists = True):
	print("CYPTrain called", crop, cfexists)
	if cfexists ==False:
		cf, dim = LoadYData(crop)
	else:
		cf = loaddf('{crop}TData.npz'.format(crop=crop))
		dim = len(cf.columns)
	cfy = cf.to_numpy()
	print("this is cf")
	q = 0
	for c in cfy[0]:
		print(q, end = '----')
		print(c, end = ',   ')
		q = q+1

	dim = dim-1
	print(dim)
	X = cf.iloc[:, 0:dim]
	Y = cf.iloc[:, dim]

	baseline_model2(X, Y, crop)

#	print("bsmodel")	
#	estimators = []
#	print("estimators")
#	estimators.append(('standardize', StandardScaler()))
#	print("estiators.append")
#	estimators.append(('mlp', KerasRegressor(build_fn=baseline_model2, epochs=2000, batch_size=16, verbose=0)))
#	print("estimators.append(mlp")
#	pipeline = Pipeline(estimators)
#	print("pipeline")
#	kfold = KFold(n_splits=10)
#	print("kfold")
#	results = cross_val_score(pipeline, X, Y, cv=kfold)
#	print("results")
#	print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def baseline_model():
	print("baselinemodel called")
	# create model
	dim = 1204
	model = Sequential()
	model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def avnormalize(arr, crop):
	print("avnormalize called")
	if crop.lower() == 'corn':
		av = 135
		high = 200
	
		#avnorm = ((arr - av) /(11/10)) + av  # originally 11/10
		#avnorm = ((np.round((arr - av), 4))**(0.98)) + av 
	print(crop.lower(), "crop is here")
	if crop.lower() == 'wheat':
		#avnorm = np.where(
		#	arr>52,
		#	arr**(99/100),
		#	arr
		#)
		#arr**(99/100)
		av = 20
		high = 130
		#avnorm = ((arr - av) /(11/10)) + av  # originally 11/10

		avnorm = arr
		
	# catch extremes
	avnorm = np.where(
		arr > high,
		arr **(0.97),
		arr
		
	)

	return avnorm



def differencematrix(arr1, truth):
	print("differencematrix called", arr1, truth)
	difmat = np.where(
		arr1>200,
		199,
		abs(arr1 - truth)

	)

	#difmat = arr1 - truth
	#difmat = np.abs(difmat)
	
	return difmat


def baseline_model2(X, Y, crop):
	epochs = 1500
	if crop.lower() == 'corn':
		epochs = 2200
	if crop.lower() == 'wheat':
		epochs = 800
	

	print("baselinemodel2 called", crop, len(X), len(Y))
	# create model
	"X = "
	"There are 4 indices per date (5) * 5 stats = 100. The first 100!"
	
	
	Ysnip = Y.iloc[:]

	"Snip Y to only the first couple inputs (columns) using pandas iloc"
	"change dim for that appropriately"
	end = 80
	start = 0
	dim = end - start
	Xsnip = X.iloc[:, start:end]
	#dim = 1204
	XX = X.iloc[:, start:end]

	model = Sequential()
	model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))

	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
	
	# train model
	history = model.fit(Xsnip, Ysnip, epochs=2200, batch_size=len(Xsnip), verbose=2)
	pyplot.plot(history.history['mse'])
	#pyplot.plot(history.history['accuracy'])
	output = model.predict(XX)
	print("y started")
	[print(yy, end = ', ') for yy in Y]
	print("y end")
	for out in output:
		print(out, end = ', ')
	print(type(output))
	print("here begins new output")
	newoutput = avnormalize(output, crop)
	for out in newoutput:
		print(out, end = ', ')
	print(type(newoutput))

	example = ((200 - 135) /(11/10)) + 135


	ypred2 = newoutput
	ypred = output
	ytrue = Y
	mse2 = mean_squared_error(ytrue, ypred2)
	mse = mean_squared_error(ytrue, ypred)
	print("mse", mse)
	print("mse2", mse2)

	#print(ypred)
	#print(ytrue.array)
	ytrue = ytrue.to_numpy()

	if crop.lower() == 'corn':
		r = 96
	if crop.lower() == 'wheat':
		r = 73
	for i in range(r):
		print(abs(ypred[i] - ytrue[i]), end = ', ')
	#print(len(difmat))
	print("begins her")
	for i in range(r):
		print(abs(ypred2[i] - ytrue[i]), end = ', ')
	print("end?")

	
	model.save('models/cropyield{crop}true.HDF5'.format(crop=crop.lower()))

	#print("difmat2", difmat)
	#pyplot.show()

	return model

CYPTrain('Corn', cfexists=True)




# remember that this predicts BU/ACRE NOT TOTAL
"Try cutting and adding parameters"
"COMPLETED add in function to convert Bu/ACRE"
"Possibly start gathering even more data, automate please!"
"COMPLETED Put some manual analytics that help push or cutoff values that stray from the average -- for example there are no true values above 200, and functionally no values below 70"
"Give error bars, understand general margin of error"

def saveML():
	None
	return None
