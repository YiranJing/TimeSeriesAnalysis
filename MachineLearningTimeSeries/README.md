## QBUS3830(Advanced Analytics) Group Assignment
### Reducing Power Supply Costs in South Australia using Statistical Time-Series and ML Methods

November, 2018

### Author
- Rhys Kilian 
- Yiran Jing
- Harry Nicol

#### DataSet description

- time period: 2016-05-01 - 2017-04-30

- freq: 30 min (48 records /per day)

- Explanatory Dependent Variable: Power demand (half hour/records)

- Regressor/Exogenous Variable: 1. price 2. weather/Temperature 


#### Business objective
South Australia short run power demand forecast

### Summary

- result **Report.pdf**

## Content within this Notebook:
### Data Cleaning
 
 
### EDA 
 - Time series decomposition
 - Seasonal plot
 
### Stationary Transformation
 - AutoCorrelation and Partial AutoCorrelation Plot for Stationary checking
 - First difference and Seasonal difference
### Modeling (Automatic selection by AIC)
- Nnive Random Walk - One setp ahead **Baseline**
- Seasonal RW - One day before **Baseline**
- Seasonal RW - One week before **Baseline**





 - ARIMA
 - Holt-winters Exponential Smoothing
### Residual diagnostics 
 - ACF and PACF plots (Residual vs Time)
 - Residual distribution (histogram and QQ plot)
### Performance Measures
 - RMSE one day, one week and six weeks ahead
 How well the model perfoms on data not used in estimation, based on test dataset
##### 1. Point forecast (Best guess) measure
- MAPE 
Recall that the mean precentage error is given by $$p_t = 100*(|(y_t - \hat{y_t})|/y_t)$$
Compared to RMSE, it has advantage of being **scale-independent**, but its disadvantage of being infinite or undefined if $y_t = 0 $ for any interested period t, and has extreme value when any $y_t$ close to 0.
- MAE
##### 2. Interval Forecast measure
- Bootstrap CI



### Forecast (Statistical Model)
- Real time forecast by Rolling Window (one step ahead)
- Real time forecast by expanding Window (one step ahead)

 

 
