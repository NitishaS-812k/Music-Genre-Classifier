"""This file produces the X and Y matrices for training and testing arrays and preprocesses them """
#required imports
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

#loading datafile
data = pd.read_csv('data.csv').values

#create dictionary for genres
genre_dict ={
    "'Electronic'" : 0,
    "'Pop'" : 1,
    "'Experimental'" :2,
    "'Industrial'" :3,
    "'World'": 4,
    "'Latin'":5,
    "'HipHop'": 6,
    "'Rap'" : 7,
    "'ElectroHouse'" : 8,
    "'Folk'" : 9,
    "'Rock'" : 10,
    "'Reggae'" : 11,
    "'Lo-fi'" :12,
    "'Instrumental'" : 13,
}

#shuffle data
data = shuffle(data)

#splicing unnnecessary columns
data = data[:,1:27]

#getting columns and features
genre_list = data[:,-1]
features = data[:,:-1]

#creating y vector
genre_list = [genre_dict.get(e, '') for e in genre_list]

y = []
for i in genre_list:
    k = [0]*len(genre_dict)
    k[i] = 1
    y.append(k)   

#dividing into training and test arrays
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size= 0.1)
X_train, X_val, y_train, y_val = train_test_split(features, y, test_size = 0.1)

#normalizing features
scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_val_scaled = scaler.fit_transform(X_val)

#function to return values when called
def return_when_called():
    return X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val