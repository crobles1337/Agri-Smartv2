import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from numpy import genfromtxt


def DateToOrder(datecsv, csvpath):
  mydate = genfromtxt(datecsv, delimiter=',') # gets np array
  #create dict 
  datedict = dict()
  for i in range (mydate[0]):
    datedict[mydate[0][i]] = mydate[1][i]
  print(datedict)
  
  mycsv = genfromtxt(csvpath, delimiter=',') #
  order = np.where(
    mycsv in datedict.keys,
    datedict[mycsv],
    mycsv
  )
  print(order)


def LinearReg(csvpath):
    data = pd.read_csv(csvpath)  # load data set
    print(data)
    newdata = data.drop(["Date	"], axis=1)
    dataprcp = data['Precipitation (mm)\t']
    index = dataprcp.index
    dataprcp = dataprcp.dropna()

    print(dataprcp.index)
    print(dataprcp)
    X = dataprcp.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = index.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    print(Y_pred)
    print(type(Y_pred))
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

    # impute
#    data = data.where(
#        data!=('NaN' or 'None'),
#        #some index such as Y_pred[data.index]

  #  )
  #  for l, v in data.items(): # however you iterate through values in pd
  #      if (v==None) or (dat=='nan'): # check what these actually are
            
                


    # save new csv
csvpath = 'CSVFiles\PracticeCSV.csv'
datecsv = 'CSVFiles/2020Dates.csv'
LinearReg(csvpath=csvpath)

#DateToOrder(datecsv, csvpath)