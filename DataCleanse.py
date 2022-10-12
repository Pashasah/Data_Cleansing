# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:39:21 2022

@author: sp7012
"""

#Clearing all the previously stored variables and data
from IPython import get_ipython
get_ipython().magic('reset -sf')


#Import required libraries for reading .csv files. plotting, averaging  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
 

#Importing LabelEncoder object from sklearn.preprocessing
#Import Encoder to convert non-numerical data to numerical data
from sklearn.preprocessing import LabelEncoder  

#classification training-set spliter and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


#Importing  five classifier models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


#Read the dataset  from .csv file ...
#...including numerical, categorical and binary values
Data=pd.read_csv('Faktura.csv',encoding='unicode_escape')
#Data cleansing
Data.drop_duplicates(keep='first')
#Display the first five rows to show how the labels and values look like
print(Data.head())
#Data cleansing
missing=Data.isnull()
#Counting the number of instances in each class
Data.groupby('Targets').count()

DataFil=Data.groupby('Targets').filter(lambda x : len(x)>1500) # Filtering out the classes with less than 1500 instances
DataFil.groupby('Targets').count() #Counting the number of instances in each class
FilTeredDaTa=DataFil.groupby('Targets').count()
DataFil.to_csv('Filtered.csv')
#Check for missing NAN values
(Data.isnull().sum())  #Counting the number of missing data
(DataFil.shape) 
print("Number of Classes", FilTeredDaTa.shape[0])
print("Number of Instances", DataFil.shape[0])

#Check data types (before runnign the Encoder)
Data.dtypes
