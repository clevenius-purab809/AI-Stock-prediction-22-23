import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import yfinance as yf
yf.pdr_override()

st.set_page_config(layout="wide")
st.title('Stock Price Prediction')
st.subheader('By Purab, Aditya, Ashmit')
st.text('We made this app for our AI project. This web app can predict prices of all Indian and U.S stocks as well as most of other international stocks.You just have to')
st.text('enter the stock ticker and it will show the summary from 2010 to 2022 as well as the 100 days moving average and 200 days moving average.')
st.text('A moving average (MA) is a stock indicator commonly used in technical analysis, used to help smooth out price data by creating a constantly updated average')
st.text('price. A rising moving average indicates that the security is in an uptrend, while a declining moving average indicates a downtrend. You can customise the')
st.text('width using settings on top right corner.')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
                            
y_symbols = [user_input]
startdate = datetime(2010,1,1)
enddate = datetime(2022,12,31)


df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)

#We'll describe the data now
st.subheader('Data from 2010-2022')
st.write(df.describe())


#Let's give user some visuals
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,4.5))
plt.plot(df.Close, 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,4.5))
plt.plot(ma100, 'r', label='MA100')
plt.plot(df.Close, 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,4.5))
plt.plot(ma100, 'r', label='MA100')
plt.plot(ma200, 'g', label='MA200')
plt.plot(df.Close, 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)



#Here our team splitted the data into train and test

data_training = pd.DataFrame(df['Close'] [0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'] [int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Now we load the model we made previously using keras
model = load_model('purab_model.h5')


#Testing phase

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []


for i in range(100, input_data.shape[0]) :
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor



#Final predictions op

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 4.5))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xticks([])
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
