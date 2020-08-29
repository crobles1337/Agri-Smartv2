import os
import numpy as np
import csv
import pandas as pd


# for fixing this, see if it works better if only inputs are ndres = [np.average(ndre), np.quantile(ndre, .25), np.quantile(ndre, .5), np.quantile(ndre, .75)] as opposed to weather + soil


def inputtoCY(userpath, scpath, sspath, supath, vpath):
    c = 0
    emptycount = 0





def collectCYinput(hwpath, scpath, sspath, supath, nppath):
    c = 0
    emptycount = 0
	
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
							if 'HistWeather.csv' in listdir:
								print("truehweathercsv")
								hw = genfromtxt(os.path.join(p2, 'HistWeather.csv'), delimiter=',', skip_header=1, usecols=(1,2,3,4,5))
								Fhw = hw.flatten('F')
								tarray.extend(Fhw)
								hwlist.append(len(Fhw))
								# extract and FLATTEN
							if 'Stress_Count.csv' in listdir:
								print("stresscounttrue")
								sc = genfromtxt(os.path.join(p2, 'Stress_Count.csv'), delimiter=',', skip_header=1)
								Fsc = sc.flatten('F')
								tarray.extend(Fsc)
								sclist.append(len(Fsc))
							if 'SoilStats.npz' in listdir:
								print("soilstatstrue")
								ss0 = load(os.path.join(p2, 'SoilStats.npz'))
								ss = ss0['arr_0']
								Fss = ss.flatten('F')
								tarray.extend(Fss)
								sslist.append(len(Fss))
							if 'SoilUncertainty.npz' in listdir:
								print("soiluncertaintytrue")
								su0 = load(os.path.join(p2, 'SoilUncertainty.npz'))
								su = su0['arr_0']
								Fsu = su.flatten('F')
								tarray.extend(Fsu)
								sulist.append(len(Fsu))
					
							print(len(tarray), "length of tarray")
							#print(tarray.shape)
							"currently the tarray is empty??"
				
							#print(tarray.shape, "shape of tarray")
							tarray.append(float(yval))
						else: 
							print("CANNOT FIND WEATHERSTATS")
							wv = False

						if wv == True:
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

	savedf(df=cf, crop = crop)
	return cf, dim