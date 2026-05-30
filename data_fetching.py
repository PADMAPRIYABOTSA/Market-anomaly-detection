#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install yfinance pandas numpy scikit-learn matplotlib plotly')


# In[3]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df=yf.download("SPY",period="1y",interval="1h",auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns=df.columns.droplevel(1)
print(f"shape:{df.shape}")
print(df.head(5))


# In[7]:


#plot SPY price and volumne over last year
fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
axes[0].plot(df.index, df['Close'], color='#1f4788',linewidth=0.8)
axes[0].set_title('SPY Price -1 year Hourly', fontsize=13)
axes[0].set_ylabel('Price (USD')

axes[1].bar(df.index,df['Volume'],color='#2e75b6',alpha=0.6,width=0.02)
axes[1].set_title('Volume',fontsize=13)
axes[1].set_ylabel('Volume')

plt.tight_layout()
plt.savefig('df_png',dpi=150, bbox_inches='tight')
plt.show()


# In[8]:


df.to_parquet("data.parquet")


# In[ ]:




