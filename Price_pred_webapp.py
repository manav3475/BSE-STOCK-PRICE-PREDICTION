import streamlit as st
from classFile import ARIMA_model
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split


def main():
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Stock Price Prediction using ARIMA</h2>
    </div><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['MARUTI','RELIANCE','TIPS']
    ar = ARIMA_model()
    option=st.sidebar.selectbox('Which company would you like to choose?',activities)
    if option=='MARUTI':
        df_weekly = pd.read_csv('C:\\Users\\Admin\\Desktop\\Summary_Project\\Company Stock Data\\Sample_MARUTI.csv',header=0,index_col='timestamp',parse_dates=True,squeeze=True)
    elif option=='RELIANCE':
        df_weekly = pd.read_csv('C:\\Users\\Admin\\Desktop\\Summary_Project\\Company Stock Data\\Sample_RELIANCE.csv',header=0,index_col='timestamp',parse_dates=True,squeeze=True)
    elif option=='TIPS':
        df_weekly = pd.read_csv('C:\\Users\\Admin\\Desktop\\Summary_Project\\Company Stock Data\\Sample_TIPS.csv',header=0,index_col='timestamp',parse_dates=True,squeeze=True)
         
    #st.subheader(option)
    train,test = train_test_split(df_weekly, test_size=0.2, shuffle = False)
    d=st.slider('Select No. of Weeks for forecasting', 1, len(test))
    if st.button('Forecast'):    
        if option=='MARUTI':
            pred, score = ar.model_predict_MARUTI(df_weekly)
            predictions = ar.model_forecast_MARUTI(df_weekly,d)
        elif option=='RELIANCE':
            pred, score = ar.model_predict_RELIANCE(df_weekly)
            predictions = ar.model_forecast_RELIANCE(df_weekly,d)
        elif option=='TIPS':
            pred, score = ar.model_predict_TIPS(df_weekly)
            predictions = ar.model_forecast_TIPS(df_weekly,d)
        #pred, test = ar.fun(df_weekly,d)
        plt.figure(figsize=(15,10))
        pred.plot(legend=True)
        test['close'].rename('Test set').plot(legend=True,c='purple')
        st.pyplot(plt)
        st.success('Model Accuracy: {:f} %'.format(score))
        st.dataframe(predictions)

                   
if __name__=='__main__':
                   main()
