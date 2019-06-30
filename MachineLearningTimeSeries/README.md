## QBUS3830(Advanced Analytics) Group Assignment
### Reducing Power Supply Costs in South Australia using Statistical Time-Series and ML Methods

November, 2018

### Author
- Rhys Kilian 
- Yiran Jing
- Harry Nicol

### Executive Summary
Our neural network model forecast for South Australian power demand was able to outperform the industry standard model by 11.12%. This project used very short-term load forecasting methods to improve the one step ahead (30 minute) demand forecast for the state of South Australia. In order to reach this optimal model we added Bureau of Meteorology Adelaide weather data as an additional variable on top of variables from the energy market operator. Using the 30 minute time series data for the input variables price, demand and temperature we considered traditional time series models, machine learning methods and benchmark models.

Through EDA we identified a strong daily and weekly pattern for demand which was utilised in auto regressive models and linear models. During feature engineering we created dummy variables for blackouts, heatwaves and season. The optimal neural network was identified through a randomised grid search cross validation of hyperparameters. Combination forecasts were considered but could not beat the performance of the neural network model. The expansion of input variables to include weather variables allows our neural network to be so successful that it could save millions of dollars for energy providers. By creating more accurate demand forecasts there will be less supply induced blackouts in South Australia allowing for a more reliable and less costly energy market.


- result [Final Report](TimeSeriesAnalysis/master/MachineLearningTimeSeries/Report.pdf)

