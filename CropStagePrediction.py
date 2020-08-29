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
import os
import pandas as pd
from keras.metrics import binary_accuracy
import tensorflow as tf
from matplotlib import pyplot
from keras.utils import to_categorical
# IMPORT NPZ LOAD

#General concept, organize images by date and then use SAVI and NDRE to predict if its 1-5 image throughout the year. I can roughly translate this
# NDRE we have
# I could just use NDRE and NDVI for this, save myself so much time, and memory
# SAVI requires NIR and Red so we can use it, I just have not yet extracted it.

"DECIDED I WILL USE NDVI AND NDRE FOR THIS"

# organize data as simply 9 values, first 6 are mean, 25 percentile, median, 75 percentile for ndvi and ndre, and then 9th is "stage"
# it should do this pretty roughly
# input as simple classification (NOT REGRESSION) between 4 (or 5 stages)
#

def LoadSData(crop):
    print("LoadSData called", crop)
    # create a panda data set that we will be appending to
    columns = 9
    count = 0
    df = pd.DataFrame(columns=range(columns))
    mpath = os.path.join(crop, 'data/train')
    for item in os.listdir(mpath):
        p1 = os.path.join(mpath, item)
        for fold in os.listdir(p1): # were at county level now
            p2 = os.path.join(p1, fold)
            if os.path.isdir(p2):
                for date in os.listdir(p2):
                    if date != 'YIELD.txt':
                        tlist = []
                        indexreal = True
                        p3 = os.path.join(p2, date)
                        tarpass = True
                        if os.path.isdir(p3):
                            for index in os.listdir(p3):
                                print(index)
                                if index == 'NDRE.npz':
                                    print("truendre")
                                    l = load(os.path.join(p3, 'NDRE.npz'))
                                    l = l['arr_0']
                                    if l.size!=0:
                                        ndre = [np.average(l), np.quantile(l, .25), np.quantile(l, .5), np.quantile(l, .75)]
                                        tlist.extend(ndre)
                                    #indexreal = True
                                    #load
                                    else:
                                        indexreal=False
                
                                if index == 'NDVI.npz':
                                    print("truendvi")
                                    l = load(os.path.join(p3, 'NDVI.npz'))
                                    l = l['arr_0']
                                    if l.size!=0:
                                        ndvi = [np.average(l), np.quantile(l, .25), np.quantile(l, .5), np.quantile(l, .75)] 
                                        print(len(ndvi), "length of ndvi")
                                        tlist.extend(ndvi)
                                        #indexreal = True
                                    else:
                                        indexreal = False
                        else:
                            tarpass = False
                            # append to a row in the 
                        print(date, "fold")
                        if '201903' in date:
                            tlist.extend([1]) # append to the last column in the appropraite row
                        if '201904' in date:
                            tlist.extend([2]) # th
                        if '201905' in date:
                            tlist.extend([3]) # th
                        if '201906' in date:
                            tlist.extend([4]) # th
                        if '201907' in date:
                            tlist.extend([5]) # th
                        if '201908' in date:
                            tlist.extend([6]) # th
                  #      if indexreal == True:
                        print(len(tlist), "length of tlist")

                        if indexreal==True:
                            if tarpass == True:
                            
                                df.loc[count] = tlist
                                count = count+1
                                print(count)
    savedf(df, crop)
    return df, columns

def CSPTrain(crop, save = True):
    print(crop, save, "CSPTrain called")
    if save == False:
        cf, dim = LoadSData(crop)
    else:
        cf = loaddf('{crop}CStageData.npz'.format(crop=crop))
    
    dim = dim = 8
    print(dim)
    X = cf.iloc[:, 0:dim]
    Y = cf.iloc[:, dim]
    estimators = []
    new2model = newbaseline_model2(X, Y, crop)
    #newmodel = newbaseline_model(X, Y)
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print(results)


# should be multiclass
#but mean squared error is also useful



def ogbaseline_model():
	# create model
	dim = 8
	model = Sequential()
	model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def baseline_model():
    dim = 8
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model


def savedf(df, crop):
	filename = '{crop}CStageData.npz'.format(crop=crop)
	df.to_pickle(filename)
	return filename
def loaddf(filename):
	cf = pd.read_pickle(filename)
	return cf	



def newbaseline_model(X, Y):
    dim = 8
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam', metrics=[['mse', 'mae', 'mape', 'cosine', 'accuracy']])
    history = model.fit(X, Y, epochs=900, batch_size=(len(X)/4), verbose=2)
# plot metrics
    pyplot.plot(history.history['mse'])
    pyplot.plot(history.history['mae'])
    pyplot.plot(history.history['mape'])
    pyplot.plot(history.history['cosine'])
    pyplot.plot(history.history['accuracy'])
    #pyplot.show()
    
    return model

def newbaseline_model2(X, Yint, crop):
    
    start = 0
    end = 8
    Ysnip = Yint.iloc[:-16]
    
    Xsnip = X.iloc[:-16, start:end]
    X = X.iloc[:, start:end]
    dim = end - start
    #Y = to_categorical(Yint)
    Y = Yint
    #Y = Yint
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(8, input_dim=dim, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(64,  kernel_initializer='normal', activation='relu'))
    #model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(64,  kernel_initializer='normal', activation='relu'))
    
    #model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))# before it was 5
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])

    history = model.fit(Xsnip, Ysnip, epochs=800, batch_size=len(X), verbose=2)
# plot metrics

    #pyplot.plot(history.history['categorical_accuracy'])
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['mse'])
    
    
# try increasing epochs/batch sizes
    output = model.predict(X)
    #print("this is output", output)
    print("this is output")
    for out in output:
        print(out, end = ', ')
    print(type(output))
    
    print("Y starts here")
    [print(yy, end = ', ') for yy in Y]
    print("Y ends here!!")
    #print((Y.to_string().replace('\n', '')))
    #print([Y])
    #print(type(Y))
    ytrue = Y
    ypred = output
    
    ypred2 = csnormalize(output, crop)
    mse = mean_squared_error(ytrue, ypred)
    mse2 = mean_squared_error(ytrue, ypred2)
    print("mse,", mse)
    print("mse2", mse2)
    ytrue = ytrue.to_numpy()
    bias = 0
    for i in range(len(ytrue)):
        print((ypred[i] - ytrue[i]), end = ', ')# eliminated abs from both
        bias = bias + (ypred[i] - ytrue[i])
    print(bias, "this is bias")
    print(bias/(len(ytrue)))
	#print(len(difmat))
    print("begin here")
    nbias = 0
    for i in range(len(ytrue)):
        print((ypred2[i] - ytrue[i]), end = ',')
        nbias = nbias + (ypred2[i] - ytrue[i])
    print(nbias, "this is bias")
    print(nbias/(len(ytrue)))
    print("end!")


    model.save('models/cropstage{crop}v1.HDF5'.format(crop=crop))
    #pyplot.show()
    return model

def csnormalize(arr, crop):
    if crop == 'Corn':
        arr = np.where(
            arr == 0,
            0,
            arr*1.05
        )
    if crop == 'Wheat':
        None
    return arr



"y_binary = to_categorical(y_int)```"


CSPTrain('Wheat', save=True)
# og was 1, 1, ,2, 3, 4, 4
# MSE was about .95

"for tomorrow"
"README"
"Actually add shapefile/coordinate/tif extraction"
"Upload to github"
"Understand how MSE cross validation scoring is happening"
"How to score with multi-class"
"How to improve results"
"Clean up functions, code, everything"


#Standardized: -1.03 (0.46) MSE
#[-0.5959164  -0.52216208 -0.86101267 -0.80812834 -1.14755144 -2.10266247
# -0.88318952 -0.71464738 -1.59003106 -1.03631257]


def MLPath():
    None
    path = "?"
    # create path
    return path

#model.save('path/to/location')

def loadml(path):    
    model = keras.models.load_model('path/to/location')
    return model


# 2 layers, 600 epochs
#[-0.52503157],[0.4414866],[-0.71318865],[-1.6730278],[-1.5211172],[-0.3144045],[0.3673277],[-0.6784539],[-1.6130841],[-1.413795],[-0.4657688],[0.43038297],[-0.8800912],[-1.6264834],[-1.4412961],[-1.3318493]
# [-0.82982635],[0.83609056],[-0.2578547],[-1.2487471],[-1.1489604],[-0.7543454],[0.7941072],[-0.23636055],[-1.2146244],[-1.0870848],[-0.82244825],[0.8367975],[-0.33466506],[-1.2253389],[-1.1028161],[-1.2407792]

#
