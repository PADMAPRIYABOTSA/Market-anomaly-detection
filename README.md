## Market-anomaly-detection
Unsupervised anomaly detection on SPY hourly market data using Isolation Forest. Flags unusual price action, volume spikes, and range explosions. For automatic flagging of data issues
# Description
this project builds an automated anomaly detection system for financial market data. Given one year of hourly SPY price and volume data, it learns what "normal" market behaviour looks like and flags the hours that deviate meaningfully from that baseline
I used Isolation Forest, an unsupervised machine learning algorithm that works by randomly partitioning the feature space
The model assigns each data point an anomaly score: more negative means more anomalous. I set the contamination threshold at 2%, meaning I expect roughly 1 in 50 hours to be flagged — a reasonable assumption for liquid equity market data under normal conditions

# Features
log_return -> How much did price move this hour?
return_zscore -> How unusual is that move relative to recent history?
range_ratio -> How wide is the high-low range relative to close?
range_zscore -> How unusual is that range relative to recent typical ranges?
volume_ratio -> How does this hour's volume compare to the 20-hour average?
volume_zscore -> How unusual is that volume ratio?
price_vs_ma -> How far is price from its 20-hour moving average?

# Results:
Running on 1,707 hourly observations across one year of SPY data:
anomalies detected=35(2.1%)

# Anomaly vs Normal
Price    log_return  volume_ratio  range_ratio  return_zscore
Normal        0.000         0.976        0.003          0.001
Anomaly      -0.004         2.416        0.009         -1.291

# Anomaly timing:
by hour of day:
Datetime
9     26
10     2
11     2
12     1
14     4
