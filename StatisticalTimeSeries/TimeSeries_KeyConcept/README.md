# Key Concepts for classical time series model 

Author: Yiran Jing
Date: 2019-07-31


## Stationary and Non-stationary Time Series
Stationary is an aspect of a sequence result, not for single point. The most commonly used term is Weak stationary:
<img width="600" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/picture.png">

Where h is the time horizon and g(h) is the function related to h only. We say it is ‘weak’ stationary because it only needs the mean and the variance exist and consistent overtime (only focus on the first and second moments of stochastic process). 

### When stationary process is important
Most statistical forecasting methods, such as ARIMA, Exponential smoothing, are based on the assumption that the time series can be rendered approximately stationary. 

We do not need to worry about stationary in ML methods, such Neural Networks model and DeepAR+ model. Since ML model for time series forecast is still cross-sectional based. 

## Difference-stationary transformation
The **first difference** of a time series is the series of changes from one period to the next. If Yt denotes the value of the time series Y at period t, then the first difference of Y at period t is equal to Yt-Yt-1. 

Using the monthly based data. let t_1 be observation in July 2018, t_0 be observation in June 2018. let t_13 be observation in July 2019, t_12 be observation in June 2019.
The main strategies to stationary time series data are: 
1.	First difference/ One-month difference = observation_(t_1)- observation_(t_0)
2.	One-year difference = observation_(t_13)-observation_(t_12) 
3. One-year plus One-month difference = [observation_(t_13)-observation_(t_12)]-[observation_(t_1)- observation_(t_0)]

#### One unit root:
- I(0): A stationary time series process, we do not need to difference to make it stationary.
- I(1)(**One unit root**): A time series is I(1) is said to be a difference-stationary process if it can get stationary after first difference
- I(D): It talks about the number of times we need to difference the time series in order to get a stationary data.

### Stationary Testing
#### ACF and PACF plot
Autocorrelation function (ACF) plot displays the autocorrelation for a range of lags.
The autocorrelation of the order k is:
<img width="600" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%208.png">

#### Augmented Dickey fuller Test
Dickey Fuller distribution is an asymptotic distribution of t-statistics under null hypothesis that a unit root is present in time series. And the alternative hypothesis is stationary time series. Thus, at the 1% level, we can say time series is stationary if p value is less than 0.01. 

### Why Differencing works
There are many ways a time series can be non-stationary. The time series pattern, including trend, seasonal pattern and cyclic, is the most common reason of non-stationary data.  Differencing dataset can eliminate time series patterns in time series data. Here I use an example of being trend stationary, which means that the time series is stationary after remove the trend pattern. 

- The first difference can be adjusted to suit the specific temporal structure and the trend (See example 2).
- For time series with a seasonal component, the lag may be expected to be the period (width) of the seasonality (see example 1).
- Some complexity case needs the combination of first difference and k-lag difference (See example 3).

#### Example 1: Eliminating Daily pattern by 48th-differencing
The following plot is ACF and PACF plot of half-hour-based time series data
<img width="900" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%201.png">
We can see daily pattern in every 24 hours. Thus, we use 48th-differnece to eliminate this daily pattern.

#### Example 2: Eliminating trend by first differencing
The original data shows a clear linear trend in the data:
<img width="600" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%202.png">

And after the first difference: 

<img width="500" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%203.png">

#### Example 3: Eliminating Year trend and Monthly pattern 
Using monthly-based data. Suppose we published a new sell strategy in July 2019, and we want to know if this new strategy help our business. The data information: observation_(t_1 )=500, observation_(t_0 ) =400, observation_(t_13 )=600, observation_(t_12 )=500. 
At the first blush, we might believe that we have a very good sell strategy since there are some sell increase in July 2019, compared to last year and last month. But is it true?  Let’s do one-year difference and one-month difference:

Year and month difference=(observation_(t_13)-observation_(t_12))-(observation_(t_1)- observation_(t_0)) 
= (600-500) – (500 - 400) = 0

We get 0 after differencing. Thus, the true story behind these observations is that this new sell strategy does not help the sell increase, after we remove the monthly trend and the year trend. 


#### Limitations of differencing:
1.	We need to transform back after model forecast. Since our model is based on the difference, rather than original value. 
2.	We will lose some information after differencing. See the table below. It is not a problem if we have many hundreds or thousands of observations. 
<img width="400" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%204.png">

## Time Series Model Diagnostics (Used for both Classical model and ML model)
#### Residual plot
The presence of patterns in the time series of residuals may suggest assumption violations and the need for alternative models. For example, see the following first plot. And the pattern dispeared in the residual plot of improved model (second plot.)

<img width="400" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%205.png">
<img width="400" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%206.png">

#### 2.	Residual ACF plot
Well specified models should lead to small and insignificant sample autocorrelations, consistent with a white noise process. For example, if we find clear pattern in the Residual ACF plot of ARIMA model(see plot below), it means that the ARIMA model we currenly choice performs really bad.
<img width="600" alt="Screen Shot 2019-06-29 at 9 29 35 pm" src="https://github.com/YiranJing/TimeSeriesAnalysis/blob/master/StatisticalTimeSeries/TimeSeries_KeyConcept/sources/Picture%207.png">















