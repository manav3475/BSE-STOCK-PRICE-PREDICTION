from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from math import sqrt
from datetime import date, timedelta

class ARIMA_model:
    def model_predict_TIPS(self,df):
        #df["lag1"] = df["close"].shift(1)
        #df = df.sort_values(by=['timestamp'])
        #train,test = df[1:df.shape[0]-177],df[df.shape[0]-177:]
        train,test = train_test_split(df, test_size=0.2, shuffle = False)
        model=ARIMA(df["close"],order=(2,3,10))
        model_fit=model.fit()
        start=len(train)
        end=len(train)+len(test)-1
        prediction = model_fit.predict(start=start,end=end,dynamic=True)
        rmse = 100-(sqrt(mean_absolute_error(test['close'],prediction)))
        return prediction, rmse

    def model_predict_RELIANCE(self,df):
        #df["lag1"] = df["close"].shift(1)
        #df = df.sort_values(by=['timestamp'])
        #train,test = df[1:df.shape[0]-177],df[df.shape[0]-177:]
        train,test = train_test_split(df, test_size=0.2, shuffle = False)
        model=ARIMA(df["close"],order=(1,3,3))
        model_fit=model.fit()
        start=len(train)
        end=len(train)+len(test)-1
        prediction = model_fit.predict(start=start,end=end,dynamic=True)
        rmse = 100-(sqrt(mean_absolute_error(test['close'],prediction)))
        return prediction, rmse

    def model_predict_MARUTI(self,df):
        #df["lag1"] = df["close"].shift(1)
        #df = df.sort_values(by=['timestamp'])
        #train,test = df[1:df.shape[0]-177],df[df.shape[0]-177:]
        train,test = train_test_split(df, test_size=0.2, shuffle = False)
        model=ARIMA(df["close"],order=(2,3,3))
        model_fit=model.fit()
        start=len(train)
        end=len(train)+len(test)-1
        prediction = model_fit.predict(start=start,end=end,dynamic=True)
        rmse = 100-(sqrt(mean_absolute_error(test['close'],prediction)))
        return prediction, rmse

    def model_forecast_TIPS(self,df,num):
        train, test = train_test_split(df, test_size=0.2, shuffle = False)
        pred = []
        # walk-forward validation
        for t in range(num):
            model = ARIMA(train['close'], order=(2,3,10))
            model_fit = model.fit()
            output = model_fit.forecast()
            l = list(output)
            yhat = l[0]
            pred.append(yhat)

        # To generate weekly date from user's choice
        lst =[]
        sdate = test.index[-1]
        for i in range(num):
            d = sdate + timedelta(days=7)
            lst.append(d)
            sdate = d

        predictions = pd.DataFrame(pred, columns=['Prediction'])
        predictions.index = lst
        return predictions
        
    def model_forecast_MARUTI(self,df,num):
        train, test = train_test_split(df, test_size=0.2, shuffle = False)
        pred = []
        # walk-forward validation
        for t in range(num):
            model = ARIMA(train['close'], order=(2,3,3))
            model_fit = model.fit()
            output = model_fit.forecast()
            l = list(output)
            yhat = l[0]
            pred.append(yhat)

        # To generate weekly date from user's choice
        lst =[]
        sdate = test.index[-1]
        for i in range(num):
            d = sdate + timedelta(days=7)
            lst.append(d)
            sdate = d

        predictions = pd.DataFrame(pred, columns=['Prediction'])
        predictions.index= lst
        return predictions
        
    def model_forecast_RELIANCE(self,df,num):
        train, test = train_test_split(df, test_size=0.2, shuffle = False)
        pred = []
        # walk-forward validation
        for t in range(num):
            model = ARIMA(train['close'], order=(1,3,3))
            model_fit = model.fit()
            output = model_fit.forecast()
            l = list(output)
            yhat = l[0]
            pred.append(yhat)

        # To generate weekly date from user's choice
        lst =[]
        sdate = test.index[-1]
        for i in range(num):
            d = sdate + timedelta(days=7)
            lst.append(d)
            sdate = d

        predictions = pd.DataFrame(pred, columns=['Prediction'])
        predictions.index= lst
        return predictions