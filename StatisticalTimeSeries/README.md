### QBUS2820TaskTwo
Time Series Forecating- Predictive Analytics

Semester 2, 2017 


#### DataSet description
Over 3000 Rossmann drug stores in 7 European countries. 
#### Business objective
Forecast six weeks daily sales for several stores by developing a univariate forecasting model.

### Summary
The objective of this forecast is to ensure that store manager could make an effective staff arrangement that enhances the productivity and motivation. The time series forecasting relies on basic understanding of time series composition and model selection. Therefore, we perform EDA to understand the pattern with the time-varying data and select ARIMA and Exponential moving average method as potential choices to formulate robust model.
- result **Report.pdf**

## Content within this Notebook:
### Data Cleaning
 - Rescale the series to be in millions of rides, to facilitate the intepretation and avoid possible numerical problems.
 - Rrop sunday since store closed
 - we also clean data when open=0
 - and after our check , we decide to remove 180 stores, with imcompleted information
### EDA 
 - Time series decomposition
 - Seasonal plot
### Stationary Transformation
 - AutoCorrelation and Partial AutoCorrelation Plot for Stationary checking
 - First difference and Seasonal difference
### Modeling (Automatic selection by AIC)
 - ARIMA
 - Holt-winters Exponential Smoothing
### Residual diagnostics 
 - ACF and PACF plots (Residual vs Time)
 - Residual distribution (histogram and QQ plot)
### Model Validation
 - RMSE one day, one week and six weeks ahead
### Forecast

 

## Author
- Yiran Jing
- Haonan Zhang
- Chenxi Zhou 
- Yunfeng Guo
