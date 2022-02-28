import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from autots import AutoTS
import os

ticker = 'BTC-USD'

sns.set()
plt.style.use('seaborn-whitegrid')

df = yf.download(
tickers = ticker,
period ='7d',
interval = "1m",
)

df.to_csv(ticker+'_price.csv')

data = pd.read_csv(ticker+"_price.csv")

print("Shape of Dataset is: ",data.shape,"\n")
print(data.head())

data.dropna()
plt.figure(figsize=(10, 4))
plt.title(ticker+"Price INR")
plt.xlabel("Date")
plt.ylabel("Close")
plt.xticks([])
plt.plot(data["Datetime"],data["Close"])
plt.show()

model = AutoTS(forecast_length=4, frequency='infer', ensemble=['horizontal-min','horizontal-max'],n_jobs='auto',max_generations=20,transformer_list='all')

if os.path.isfile('./'+ticker+'_model.json'): 
    print("found model file")
    model.import_template(filename='./'+ticker+'_model.json',method='only')
else :
    print("not found model file")

model = model.fit(data, date_col='Datetime', value_col='Close', id_col=None)

print('model fited !')

model.export_template('./'+ticker+'_model.json', models='best',include_results=True)

# show data
prediction = model.predict()
forecast = prediction.forecast
print(ticker+"Price Prediction")
print(forecast)

forecast.to_csv('Price_Prediction.csv')

plt.figure(figsize=(10, 4))
plt.title(ticker + " Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(forecast["Close"])
plt.show()
