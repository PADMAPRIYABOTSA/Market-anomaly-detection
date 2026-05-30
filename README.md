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

## Anomaly vs Normal — Average Characteristics

| | Log Return | Volume Ratio | Range Ratio | Return Z-Score |
|---|---|---|---|---|
| **Normal** | 0.000 | 0.976 | 0.003 | 0.001 |
| **Anomaly** | -0.004 | 2.416 | 0.009 | -1.291 |

## Anomaly Timing — By Hour of Day (ET)

| Hour | Anomalies |
|---|---|
| 9am (market open) | 26 |
| 10am | 2 |
| 11am | 2 |
| 12pm | 1 |
| 2pm | 4 |
<img width="1512" height="600" alt="anomalies_dashboard" src="https://github.com/user-attachments/assets/d41c35c5-6f0f-4215-a425-1c242f944c98" />
# SPY Price with Anomalies:
1.SPY went from 600 around in June to 750 by May of the next year.
2.There are several anomalies bunched together at certain period. 
3.Coinciding with what looks like a sharp drop and recovery in SPY price around the period, there are anomalities clustered.

# Anomaly Score:
1.Most scores are between -0.1 and 0.2
2.The deep dips below -0.1 — those are the hours the model found most unusual. Notice they appear at roughly the same times as the red X clusters in panel 1 — Sep 2025, Jan 2026, and May 2026
