import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def deg_to_rad(degrees):
    try:
        #logging.info("Converting degree to radian")
        return degrees * (np.pi/180)
    except Exception as e:
        logging.info('Exception occured while converting points from Degree to radian')
        raise CustomException(e, sys)
        

        # Function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    try:

        #logging.info('Calculating the distance using haversine formula')
        R=6371
        lat1=abs(lat1)
        lat2=abs(lat2)
        lon1=abs(lon1)
        lon2=abs(lon2)
        

        d_lat = deg_to_rad(lat2-lat1)
        d_lon = deg_to_rad(lon2-lon1)
        a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    except Exception as e:
        logging.info('Exception occured while calculating the distance')
        raise CustomException(e, sys)

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:

        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range (len(models)):
            model = list(models.values())[i]

            #Train model

            model.fit(X_train,y_train)

            # Predict Testing data

            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException( e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)