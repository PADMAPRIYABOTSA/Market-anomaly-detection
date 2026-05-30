#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np

df=pd.read_parquet("data.parquet")


# In[3]:


#log-return
df['log_return']=np.log(df['Close']/df['Close'].shift(1))
#return z-score
rolling_mean=df['log_return'].rolling(window=20).mean()
rolling_std=df['log_return'].rolling(window=20).std()
df['return_zscore']=(df['log_return']-rolling_mean)/rolling_std
#price range ratio
df['range_ratio']=(df['High']-df['Low'])/df['Close']
#range z-score
range_mean=df['range_ratio'].rolling(window=20).mean()
range_std=df['range_ratio'].rolling(window=20).std()
df['range_zscore']=(df['range_ratio']-range_mean)/range_std
#volume ratio
df['volume_ratio']=df['Volume']/df['Volume'].rolling(window=20).mean()
#volume z-score
vol_mean=df['volume_ratio'].rolling(window=20).mean()
vol_std=df['volume_ratio'].rolling(window=20).std()
df['volume_zscore']=(df['volume_ratio']-vol_mean)/vol_std
#price vs moving avergae distance
df['ma_20']=df['Close'].rolling(window=20).mean()
df['price_vs_ma']=(df['Close']-df['ma_20'])/df['ma_20']
#clean the data
df=df.dropna()
feature_cols=[
    'log_return',
    'return_zscore',
    'range_ratio',
    'range_zscore',
    'volume_ratio',
    'volume_zscore',
    'price_vs_ma'
]
x=df[feature_cols]
print(x.head(5))
print(x.describe().round(2))


# In[8]:


x.to_parquet("data_to_fit.parquet") #this is the data that contains only the features
df.to_parquet("entire_dataframe.parquet") #this contains raw columns and features


# In[ ]:




