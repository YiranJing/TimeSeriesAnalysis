## Statistical Time Series Regression 
Econometric Analysis(Ecmt2160) Time Series Notes

Semester 2, 2019
### Author: 
- Yiran Jing

### Summary
In this notes,  we focus on OLS method in time series data and thus, firstly,  we mustjudge ’correct’ model based on **Gauss-Markov Assumptions**, which are similar withcross sectional data, but actually more challenge and harder to achieve.

Unlike cross sectional data, time series have some special features: trend, cyclic, seasonality. We only consider simple trend pattern (linear and exponential trend) in this course.

To make model reliable, we emphasis **stationary** and **weak dependency**.  As they areimportant to ensure stable properties of stochastic process over time.  Also, CLT and LLN need weak dependence and stationary hold.

For the model part, we study **static model** first, which is not used for forecast, but goodat trade-off interpretation, and also the basic idea of hypothesis tests (e.g.  serial corre-lation test) and some advanced model (e.g. error correction model). **Lag distributed model** is another basic model, based on the intuition that time series always have ’lage ffect’.  For forecast purpose, we study **AR** and **VAR** model.

Given I(1) series, We talk about two potential issues in time series linear regression: **Spurious** and **cointegration**. Spurious regression is actually no relationship but we falsely believe they have, while cointegration is related to long run relationship but inequilibrium in short run.  From this point, we can realize how important to ensure stationary and weak dependent for modelling.  Otherwise, we can easily make mistake and cannot figure out.

we  discuss **hypothesis tests** for  unit  root,  serial  correlation,  cointegration,  Grangercausality.  But actually hypothesis test plays minor role in statistical inference, as they arebased on model, but all models are wrong.  And statistical significant has no relationship with practical significant
