# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from math import sqrt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QFileDialog 
from PyQt5.QtGui import QPixmap
from tensorflow.keras import Model
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # to convert it in the range of 0-1
from keras.models import Sequential # sequential
from keras.layers import Dense, LSTM, Dropout #output layer - dense - it is a layer to get te output,dropout - accuracy increases
from sklearn.metrics import mean_squared_error
from keras.layers import Bidirectional
import math
from xlrd import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

class MainWindow(QDialog):
    global crime
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("crime_screen11.ui",self)
        self.predictButton.clicked.connect(self.process)
        self.browseAfile.clicked.connect(self.gotologin)
        self.GraphBt.clicked.connect(self.gotologin1)
        #self.rmse.clicked.connect(self.rmsee)
        
    def process(self):    

        stateName = self.stateNamesCombo.currentText()
        stateName = stateName + ".xlsx"
        print(stateName)
        crime = pd.read_excel(stateName)
        crime.columns = crime.columns.str.upper()
        
        #total crime list
        crime_list = ["CRUELTY BY HUSBAND OR HIS RELATIVES", "RAPE", "INSULT TO MODESTY OF WOMEN", 'TOTAL', 'STATE', 'DOWRY DEATHS', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY', 'KIDNAPPING AND ABDUCTION',  'DISTRICT', 'YEAR']
        
        #drop all columns from data except the specific crime input by user        
        crime_name = self.crimeCombo.currentText()
        print("crime =", crime_name)
        
        for i in crime_list:
            if i != crime_name:
                crime.drop(i, axis= 1, inplace= True)
                
        d = crime
                
        print(d)
        print(d.shape)
        
        sc = MinMaxScaler(feature_range=(0,1))  #to convert to the data in the range of 0-1 (normalization of data)
        training_set_scaled = sc.fit_transform(d)
        
        #TRAIN
        X_train = []
        y_train = []
        for i in range(3,13):
            X_train.append(training_set_scaled[i-3:i,0])
            print(X_train)
            y_train.append(training_set_scaled[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
        
        #CREATE MODEL
        model = Sequential()
        model.add(Bidirectional(LSTM(25, activation='relu'), input_shape=(X_train.shape[1],1)))# denote the number of input x_....  
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.summary()
        
        #Compile Model
        model.compile(optimizer="adam", loss="mse") #adam= optimizer to control the condition of overfitting mse to find the loss(to determine the accuracy)
        model.fit(X_train,y_train,epochs=200, verbose=1) #epochs= the round of learning
        values = []
        for column in crime:
             
            # Select column contents by column
            # name using [] operator
            columnSeriesObj = crime[column]
           
            #print('Column Contents : ', columnSeriesObj.values)
            values = list(columnSeriesObj.values)
            
        
        
        
        i1 = values[len(values)-1]
        i2 = values[len(values)-2]
        i3 = values[len(values)-3]
        
        x = [i3, i2, i1]
        
        
            
        #x=([ 310,294,327])
        x_reshaped = np.reshape(x,[-1,1])
        x = sc.fit_transform(x_reshaped)
        
        x= x.reshape((1,X_train.shape[1],1))
        y = model.predict(x,verbose=0)
      
        
        
        result = sc.inverse_transform(y)
        
      
        
        self.resultEdit.setText(str(result[0][0]))
        
        if len(result)!=0:
            
            originsl_list = d.values
	
            sc = MinMaxScaler(feature_range=(0,1))  #to convert to the data in the range of 0-1 (normalization of data)
            training_set_scaled = sc.fit_transform(d)
            
            
            size = int(len(training_set_scaled)*0.6)
            train, test = training_set_scaled[0:size], training_set_scaled[size:len(training_set_scaled)]
            predictions = list()
            observed = list()
            
            X_train = []
            y_train = []
            X_train_list = []
            y_train_list = []
            
            
            for i in range(3,size):
                X_train_list.append(training_set_scaled[i-3:i,0])
                
                y_train_list.append(training_set_scaled[i,0])
            
            
            X_train, y_train = np.array(X_train_list), np.array(y_train_list)
            X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
            print(X_train)
            
            
            rolling_index=size
            
            # walk-forward validation
            for t in range(len(test)):
            
                #convert type
                model = Sequential()
                model.add(Bidirectional(LSTM(25, activation='relu'), input_shape=(X_train.shape[1],1)))# denote the number of input x_....  
                model.add(Dropout(0.2))
                model.add(Dense(units=1))
                
                model.summary()
                
                #Compile Model
                model.compile(optimizer="adam", loss="mse") #adam= optimizer to control the condition of overfitting mse to find the loss(to determine the accuracy)
                model.fit(X_train,y_train,epochs=200, verbose=1) 
                
                
                #Scaling
                i1 = training_set_scaled[rolling_index-1]
                i2 = training_set_scaled[rolling_index-2]
                i3 = training_set_scaled[rolling_index-3]
                
                x = [i3, i2, i1]
                x = np.array(x)
               
                x= x.reshape((1,X_train.shape[1],1))
                y = model.predict(x,verbose=0)
                
             
                result = sc.inverse_transform(y)   
        
                yhat = result[0][0]
                
          
                predictions.append(yhat) 
                obs=originsl_list[rolling_index-1]
              
                observed.append(obs)
                #print('predicted=%f, expected=%f' % (yhat, obs))
                
                #append for rolling forecast               
                X_train_list.append(training_set_scaled[rolling_index-3:rolling_index,0])
               
                y_train_list.append(training_set_scaled[rolling_index,0])
                
                
                X_train, y_train = np.array(X_train_list), np.array(y_train_list)
                X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
                
                rolling_index = rolling_index + 1
            
            
    
                # evaluate forecasts
                rmse = sqrt(mean_squared_error(observed, predictions))
                print('Test RMSE: %.3f' % rmse)
                plt.plot(observed)
                plt.plot(predictions, color='red')
                plt.show()
 #               self.rmse_result.setText(str(int(rmse)))
        
    def gotologin(self):
        login = AnalyserScreen()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def gotologin1(self):
        login = Graph()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)
   
class Graph(QDialog):
    def __init__(self):
        super(Graph, self).__init__()
        loadUi("crime_screen3.ui",self)
        self.GraphBack.clicked.connect(self.gotoCreate)
        
    def gotoCreate(QDialog):
         mainwindow = MainWindow()
         widget.addWidget(mainwindow)
         widget.setCurrentIndex(widget.currentIndex()+1)

class AnalyserScreen(QDialog):
    def __init__(self):
        super(AnalyserScreen, self).__init__()
        loadUi("crime_screen22.ui",self)
        self.bt4.clicked.connect(self.gotoback)
        self.predictButton.clicked.connect(self.browsefiles)
        self.bt4_2.clicked.connect(self.predict)
        
        
    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self, '(*.png, *.xmp *.xlsx *.jpg)')
        self.resultEdit1.setText(fname[0])
        path = fname[0]
        global new
        new = os.path.basename(path)

   
        
        
    def predict(self):
        pd1 = pd.read_excel(new)
        pd1.columns = pd1.columns.str.upper()
        crime =pd1[['TOTAL']]
        sc = MinMaxScaler(feature_range=(0,1))  #to convert to the data in the range of 0-1 (normalization of data)
        training_set_scaled = sc.fit_transform(crime)
        
        #TRAIN
        X_train = []
        y_train = []
        for i in range(3,13):
            X_train.append(training_set_scaled[i-3:i,0])
            print(X_train)
            y_train.append(training_set_scaled[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
        
        #CREATE MODEL
        model = Sequential()
        model.add(Bidirectional(LSTM(25, activation='relu'), input_shape=(X_train.shape[1],1)))# denote the number of input x_....  
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.summary()
        
        #Compile Model
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train,y_train,epochs=200, verbose=1)
        
        
        values = []
        for column in crime:
            columnSeriesObj = crime[column]
            values = list(columnSeriesObj.values)

        i1 = values[len(values)-1]
        i2 = values[len(values)-2]
        i3 = values[len(values)-3]
        
        x = [i3, i2, i1]

        x_reshaped = np.reshape(x,[-1,1])
        x = sc.fit_transform(x_reshaped)
        x= x.reshape((1,X_train.shape[1],1))
        y = model.predict(x,verbose=0)
        result = sc.inverse_transform(y)
        
        self.resultEdit_2.setText(str(result[0][0]))
        
        if len(result)!=0:
        

            d= crime.values
            
            originsl_list = d
    	
            sc = MinMaxScaler(feature_range=(0,1))  #to convert to the data in the range of 0-1 (normalization of data)
            training_set_scaled = sc.fit_transform(d)
            
            
            size = int(len(training_set_scaled)*0.6)
            train, test = training_set_scaled[0:size], training_set_scaled[size:len(training_set_scaled)]
            predictions = list()
            observed = list()
            
            X_train = []
            y_train = []
            X_train_list = []
            y_train_list = []
            
            
            for i in range(3,size):
                X_train_list.append(training_set_scaled[i-3:i,0])
                
                y_train_list.append(training_set_scaled[i,0])
            
            
            X_train, y_train = np.array(X_train_list), np.array(y_train_list)
            X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
            print(X_train)
            
            
            rolling_index=size
            
            # walk-forward validation
            for t in range(len(test)):
            
                #convert type
                model = Sequential()
                model.add(Bidirectional(LSTM(25, activation='relu'), input_shape=(X_train.shape[1],1)))# denote the number of input x_....  
                model.add(Dropout(0.2))
                model.add(Dense(units=1))
                
                model.summary()
                
                #Compile Model
                model.compile(optimizer="adam", loss="mse") #adam= optimizer to control the condition of overfitting mse to find the loss(to determine the accuracy)
                model.fit(X_train,y_train,epochs=200, verbose=1) 
                
                
                #Scaling
                i1 = training_set_scaled[rolling_index-1]
                i2 = training_set_scaled[rolling_index-2]
                i3 = training_set_scaled[rolling_index-3]
                
                x = [i3, i2, i1]
                x = np.array(x)
               
                x= x.reshape((1,X_train.shape[1],1))
                y = model.predict(x,verbose=0)
                
             
                result = sc.inverse_transform(y)   
        
                yhat = result[0][0]
                
          
                predictions.append(yhat) 
                obs=originsl_list[rolling_index-1]
              
                observed.append(obs)
                #print('predicted=%f, expected=%f' % (yhat, obs))
                
                #append for rolling forecast
                
                X_train_list.append(training_set_scaled[rolling_index-3:rolling_index,0])
               
                y_train_list.append(training_set_scaled[rolling_index,0])
                
                
                X_train, y_train = np.array(X_train_list), np.array(y_train_list)
                X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
                
                rolling_index = rolling_index + 1
            
            
   
            # evaluate forecasts
            rmse = sqrt(mean_squared_error(observed, predictions))
            print('Test RMSE: %.3f' % rmse)
            plt.plot(observed)
            plt.plot(predictions, color='red')
            plt.show()
#            self.rmse_result2.setText(str(rmse))
               
            
    def gotoback(QDialog):
         mainwindow = MainWindow()
         widget.addWidget(mainwindow)
         widget.setCurrentIndex(widget.currentIndex()+1)

app=QApplication(sys.argv)
mainwindow = MainWindow()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(800)
widget.setFixedHeight(630)
widget.show()
sys.exit(app.exec_())





