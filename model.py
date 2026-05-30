#!/usr/bin/env python
# coding: utf-8

# In[34]:


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
x=pd.read_parquet("data_to_fit.parquet")
df=pd.read_parquet("entire_dataframe.parquet")


# In[35]:


#scaling features
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
model=IsolationForest(
    n_estimators=200, # no. of trees
    contamination=0.02, #expected % of anomalies
    random_state=48,
    n_jobs=-1 #use all cpu cores
)
model.fit(x_scaled)
df['anomaly_label']=model.predict(x_scaled) #-1=anomaly, 1=normal
df['anomaly_score']=model.decision_function(x_scaled) # mroe negative more anomalous
df['is_anomaly']=df['anomaly_label']==-1 #flagging anomalies
n_anomalies=df['is_anomaly'].sum()
print(n_anomalies)
df['is_anomaly'].value_counts()


# In[36]:


df.head(10)


# In[37]:


normal=df[~df['is_anomaly']]
normal.head(5)


# In[ ]:





# In[38]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[39]:


fig=make_subplots(
    rows=2,cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "SPY Price- anomalies highlighted in red",
        "Anomaly score over time (more negative = more unsual)",
    ],
    vertical_spacing=0.1,
    row_heights=[0.65,0.35]
)

normal=df[~df['is_anomaly']]
anomalies=df[df['is_anomaly']]
fig.add_trace(
    go.Scatter(
        x=normal.index,y=normal['Close'],
        mode='lines',name='Normal',
        line=dict(color='#1f4788',width=1)
    ), row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=anomalies.index, y=anomalies['Close'],
        mode='markers',name='Anomaly',
        marker=dict(color='red',size=6,symbol='x')
    ),row=1,col=1
)
fig.add_trace(
    go.Scatter(
        x=x.index,y=df['anomaly_score'],
        mode='lines',name='Anomaly Score',
        line=dict(color='#9467bd',width=0.8)
        
    ),row=2,col=1
)
fig.update_layout(
    title=dict(
        text="Market Data Anomaly Detection- SPY Hourly Data",
        font=dict(size=16, color='#1f4788')
    ),
    height=600,
    template='plotly_white',
    hovermode='x unified',
    showlegend=True
)
fig.write_html("anomaly_dashboard.html",include_plotlyjs='cdn')
fig.show()


# In[41]:


print("Anomaly analysis report")
top_anomalies=(df[df['is_anomaly']].sort_values('anomaly_score').head(10))
print(top_anomalies[[
    'Close', 'log_return', 'volume_ratio',
    'range_ratio', 'anomaly_score'
]].mean().round(4).to_string())
print("anomay vs normal- average characteristics")
comparison=df.groupby('is_anomaly')[[
    'log_return', 'volume_ratio',
    'range_ratio', 'return_zscore'
]].mean().round(3)
comparison.index=['Normal','Anomaly']
print(comparison.to_string())
print("Anomaly timing:")
anomaly_hours=anomalies.index.hour.value_counts().sort_index()
print("by hour of day:")
print(anomaly_hours.to_string())


# In[42]:


print(f"Total rows: {len(df)}")
print(f"Anomalies detected: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(x)*100:.1f}%)")


# In[ ]:




