# SARIMA

SARIMA Code:

Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
Importing dataset
df=pd.read_csv("Belfast-2015-2022-data.csv")
df.head()
Number	Date	O3	NO2	PM10	PM2.5	temp	humidity	Unnamed: 8	Unnamed: 9
0	1	14/01/2015	48	28	9	6	3.2	85.4	NaN	NaN
1	2	15/01/2015	61	17	15	7	5.0	76.1	NaN	NaN
2	3	16/01/2015	53	23	9	6	2.0	85.3	NaN	NaN
3	4	17/01/2015	51	24	7	6	1.5	90.3	NaN	NaN
4	5	18/01/2015	59	14	11	7	2.8	76.9	NaN	NaN
Calculating monthly mean between 2015-2022
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
y = df.resample('M')['O3'].mean()
print(y)
Date
2015-01-31    43.392857
2015-02-28    45.333333
2015-03-31    43.640000
2015-04-30    46.115385
2015-05-31    53.960000
                ...    
2022-08-31    44.000000
2022-09-30    47.346154
2022-10-31    42.285714
2022-11-30    40.291667
2022-12-31    40.185185
Freq: M, Name: O3, Length: 96, dtype: float64
Splitting data into train & test set
y_train=y[:len(y)-12]
y_test=y[(len(y)-12):]
y_train[-12:]
Date
2021-01-31    42.333333
2021-02-28    51.423077
2021-03-31    52.448276
2021-04-30    60.068966
2021-05-31    61.000000
2021-06-30    44.172414
2021-07-31    44.000000
2021-08-31    41.666667
2021-09-30    42.285714
2021-10-31    50.117647
2021-11-30    47.068966
2021-12-31    38.433333
Freq: M, Name: O3, dtype: float64
y_train.plot()
<AxesSubplot:xlabel='Date'>

y_test[-12:]
Date
2022-01-31    40.583333
2022-02-28    55.541667
2022-03-31    46.964286
2022-04-30    53.100000
2022-05-31    53.850000
2022-06-30    45.692308
2022-07-31    45.136364
2022-08-31    44.000000
2022-09-30    47.346154
2022-10-31    42.285714
2022-11-30    40.291667
2022-12-31    40.185185
Freq: M, Name: O3, dtype: float64
y_test.plot()
<AxesSubplot:xlabel='Date'>

Ad Fuller test to check the staionarity of the series
# Transformations may be needed
result = adfuller(y_train)
​
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
​
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
ADF Statistic: -5.070579
p-value: 0.000016
Critical Values:
	1%: -3.513
	5%: -2.897
	10%: -2.586
Its is clear that the series is indeed stationary
Autocorrelation Factor and Partial Autocorrelation Factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
​
fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(y_train, ax=ax[0], lags=20)
ax[1] = plot_pacf(y_train, ax=ax[1], lags=20)

Looking at time series decomposition to see trend, seasonality & residuals
ts_decop = sm.tsa.seasonal_decompose(y_train, model='additive')
ts_decop.plot()
plt.show()

Since we have strong seasonality we will use seasonal ARIMA instead of ARIMA
SARIMA has parameters, 3 for ARIMA and 4 for the seasonal componenet
p = d = q = range(0, 2)
​
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12,) for x in list(itertools.product(p, d, q))]
​
print('Examples of parameter combinations for seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[5]))
print('SARIMAX: {} x {}'.format(pdq[3], seasonal_pdq[5]))
Examples of parameter combinations for seasonal ARIMA...
SARIMAX: (0, 0, 1) x (0, 0, 1, 12)
SARIMAX: (0, 0, 1) x (0, 1, 0, 12)
SARIMAX: (0, 1, 0) x (0, 1, 1, 12)
SARIMAX: (0, 1, 0) x (1, 0, 0, 12)
SARIMAX: (0, 1, 0) x (1, 0, 1, 12)
SARIMAX: (0, 1, 1) x (1, 0, 1, 12)
metric_aic_dict=dict()
​
for pm in pdq:
    for pm_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(y_train,
                                             order=pm,
                                             seasonal_order=pm_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
            model_aic = model.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(pm, pm_seasonal, model_aic.aic))
            metric_aic_dict.update({(pm,pm_seasonal):model_aic.aic})
        except:
            continue
ARIMA(0, 0, 0)x(0, 0, 0, 12)12 - AIC:869.6198306654262
ARIMA(0, 0, 0)x(0, 0, 1, 12)12 - AIC:686.1546805384622
ARIMA(0, 0, 0)x(0, 1, 0, 12)12 - AIC:474.8232899490605
ARIMA(0, 0, 0)x(0, 1, 1, 12)12 - AIC:377.1285870880863
ARIMA(0, 0, 0)x(1, 0, 0, 12)12 - AIC:482.51852385919904
ARIMA(0, 0, 0)x(1, 0, 1, 12)12 - AIC:444.93459052257134
ARIMA(0, 0, 0)x(1, 1, 0, 12)12 - AIC:375.65122425416797
ARIMA(0, 0, 0)x(1, 1, 1, 12)12 - AIC:370.9623072138102
ARIMA(0, 0, 1)x(0, 0, 0, 12)12 - AIC:762.8080147807718
ARIMA(0, 0, 1)x(0, 0, 1, 12)12 - AIC:602.5706379110426
ARIMA(0, 0, 1)x(0, 1, 0, 12)12 - AIC:451.799176886463
ARIMA(0, 0, 1)x(0, 1, 1, 12)12 - AIC:357.74674085123047
ARIMA(0, 0, 1)x(1, 0, 0, 12)12 - AIC:464.96953508304466
ARIMA(0, 0, 1)x(1, 0, 1, 12)12 - AIC:429.17418392789045
ARIMA(0, 0, 1)x(1, 1, 0, 12)12 - AIC:369.8280868183521
ARIMA(0, 0, 1)x(1, 1, 1, 12)12 - AIC:357.79373901946923
ARIMA(0, 1, 0)x(0, 0, 0, 12)12 - AIC:519.6567981558159
ARIMA(0, 1, 0)x(0, 0, 1, 12)12 - AIC:444.22733677762824
ARIMA(0, 1, 0)x(0, 1, 0, 12)12 - AIC:459.7931402248957
ARIMA(0, 1, 0)x(0, 1, 1, 12)12 - AIC:368.9658636247979
ARIMA(0, 1, 0)x(1, 0, 0, 12)12 - AIC:445.92092058910504
ARIMA(0, 1, 0)x(1, 0, 1, 12)12 - AIC:442.09384556578357
ARIMA(0, 1, 0)x(1, 1, 0, 12)12 - AIC:381.0327714321046
ARIMA(0, 1, 0)x(1, 1, 1, 12)12 - AIC:372.72099453201406
ARIMA(0, 1, 1)x(0, 0, 0, 12)12 - AIC:516.165922651081
ARIMA(0, 1, 1)x(0, 0, 1, 12)12 - AIC:440.4675299448248
ARIMA(0, 1, 1)x(0, 1, 0, 12)12 - AIC:449.0583761913741
ARIMA(0, 1, 1)x(0, 1, 1, 12)12 - AIC:357.1195058615217
ARIMA(0, 1, 1)x(1, 0, 0, 12)12 - AIC:446.75175372502457
ARIMA(0, 1, 1)x(1, 0, 1, 12)12 - AIC:432.9347988653995
ARIMA(0, 1, 1)x(1, 1, 0, 12)12 - AIC:369.48649563926796
ARIMA(0, 1, 1)x(1, 1, 1, 12)12 - AIC:357.1842424403133
ARIMA(1, 0, 0)x(0, 0, 0, 12)12 - AIC:526.6470324666358
ARIMA(1, 0, 0)x(0, 0, 1, 12)12 - AIC:451.33082188873937
ARIMA(1, 0, 0)x(0, 1, 0, 12)12 - AIC:450.19687300224626
ARIMA(1, 0, 0)x(0, 1, 1, 12)12 - AIC:363.03916363394023
ARIMA(1, 0, 0)x(1, 0, 0, 12)12 - AIC:447.1239400013018
ARIMA(1, 0, 0)x(1, 0, 1, 12)12 - AIC:433.3841774016594
ARIMA(1, 0, 0)x(1, 1, 0, 12)12 - AIC:362.9047675385179
ARIMA(1, 0, 0)x(1, 1, 1, 12)12 - AIC:364.65875486862063
ARIMA(1, 0, 1)x(0, 0, 0, 12)12 - AIC:523.052552949853
ARIMA(1, 0, 1)x(0, 0, 1, 12)12 - AIC:448.03255716119793
ARIMA(1, 0, 1)x(0, 1, 0, 12)12 - AIC:446.90273587615104
ARIMA(1, 0, 1)x(0, 1, 1, 12)12 - AIC:357.1927563081098
ARIMA(1, 0, 1)x(1, 0, 0, 12)12 - AIC:448.11849715798934
ARIMA(1, 0, 1)x(1, 0, 1, 12)12 - AIC:430.05574244758014
ARIMA(1, 0, 1)x(1, 1, 0, 12)12 - AIC:364.8850017010421
ARIMA(1, 0, 1)x(1, 1, 1, 12)12 - AIC:358.5191491570879
ARIMA(1, 1, 0)x(0, 0, 0, 12)12 - AIC:521.6060210726524
ARIMA(1, 1, 0)x(0, 0, 1, 12)12 - AIC:446.2192368269003
ARIMA(1, 1, 0)x(0, 1, 0, 12)12 - AIC:458.38926296850997
ARIMA(1, 1, 0)x(0, 1, 1, 12)12 - AIC:368.4470191679689
ARIMA(1, 1, 0)x(1, 0, 0, 12)12 - AIC:441.87056997067896
ARIMA(1, 1, 0)x(1, 0, 1, 12)12 - AIC:442.6152971579603
ARIMA(1, 1, 0)x(1, 1, 0, 12)12 - AIC:369.07175041905805
ARIMA(1, 1, 0)x(1, 1, 1, 12)12 - AIC:370.4611339438927
ARIMA(1, 1, 1)x(0, 0, 0, 12)12 - AIC:504.99445549115416
ARIMA(1, 1, 1)x(0, 0, 1, 12)12 - AIC:432.3653111125824
ARIMA(1, 1, 1)x(0, 1, 0, 12)12 - AIC:443.10838205239725
ARIMA(1, 1, 1)x(0, 1, 1, 12)12 - AIC:351.92946398955104
ARIMA(1, 1, 1)x(1, 0, 0, 12)12 - AIC:433.3283092282824
ARIMA(1, 1, 1)x(1, 0, 1, 12)12 - AIC:426.93937273744064
ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:358.4566470477454
ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:353.9634331031862
{k: v for k, v in sorted(metric_aic_dict.items(), key=lambda x: x[1])}
{((1, 1, 1), (0, 1, 1, 12)): 351.92946398955104,
 ((1, 1, 1), (1, 1, 1, 12)): 353.9634331031862,
 ((0, 1, 1), (0, 1, 1, 12)): 357.1195058615217,
 ((0, 1, 1), (1, 1, 1, 12)): 357.1842424403133,
 ((1, 0, 1), (0, 1, 1, 12)): 357.1927563081098,
 ((0, 0, 1), (0, 1, 1, 12)): 357.74674085123047,
 ((0, 0, 1), (1, 1, 1, 12)): 357.79373901946923,
 ((1, 1, 1), (1, 1, 0, 12)): 358.4566470477454,
 ((1, 0, 1), (1, 1, 1, 12)): 358.5191491570879,
 ((1, 0, 0), (1, 1, 0, 12)): 362.9047675385179,
 ((1, 0, 0), (0, 1, 1, 12)): 363.03916363394023,
 ((1, 0, 0), (1, 1, 1, 12)): 364.65875486862063,
 ((1, 0, 1), (1, 1, 0, 12)): 364.8850017010421,
 ((1, 1, 0), (0, 1, 1, 12)): 368.4470191679689,
 ((0, 1, 0), (0, 1, 1, 12)): 368.9658636247979,
 ((1, 1, 0), (1, 1, 0, 12)): 369.07175041905805,
 ((0, 1, 1), (1, 1, 0, 12)): 369.48649563926796,
 ((0, 0, 1), (1, 1, 0, 12)): 369.8280868183521,
 ((1, 1, 0), (1, 1, 1, 12)): 370.4611339438927,
 ((0, 0, 0), (1, 1, 1, 12)): 370.9623072138102,
 ((0, 1, 0), (1, 1, 1, 12)): 372.72099453201406,
 ((0, 0, 0), (1, 1, 0, 12)): 375.65122425416797,
 ((0, 0, 0), (0, 1, 1, 12)): 377.1285870880863,
 ((0, 1, 0), (1, 1, 0, 12)): 381.0327714321046,
 ((1, 1, 1), (1, 0, 1, 12)): 426.93937273744064,
 ((0, 0, 1), (1, 0, 1, 12)): 429.17418392789045,
 ((1, 0, 1), (1, 0, 1, 12)): 430.05574244758014,
 ((1, 1, 1), (0, 0, 1, 12)): 432.3653111125824,
 ((0, 1, 1), (1, 0, 1, 12)): 432.9347988653995,
 ((1, 1, 1), (1, 0, 0, 12)): 433.3283092282824,
 ((1, 0, 0), (1, 0, 1, 12)): 433.3841774016594,
 ((0, 1, 1), (0, 0, 1, 12)): 440.4675299448248,
 ((1, 1, 0), (1, 0, 0, 12)): 441.87056997067896,
 ((0, 1, 0), (1, 0, 1, 12)): 442.09384556578357,
 ((1, 1, 0), (1, 0, 1, 12)): 442.6152971579603,
 ((1, 1, 1), (0, 1, 0, 12)): 443.10838205239725,
 ((0, 1, 0), (0, 0, 1, 12)): 444.22733677762824,
 ((0, 0, 0), (1, 0, 1, 12)): 444.93459052257134,
 ((0, 1, 0), (1, 0, 0, 12)): 445.92092058910504,
 ((1, 1, 0), (0, 0, 1, 12)): 446.2192368269003,
 ((0, 1, 1), (1, 0, 0, 12)): 446.75175372502457,
 ((1, 0, 1), (0, 1, 0, 12)): 446.90273587615104,
 ((1, 0, 0), (1, 0, 0, 12)): 447.1239400013018,
 ((1, 0, 1), (0, 0, 1, 12)): 448.03255716119793,
 ((1, 0, 1), (1, 0, 0, 12)): 448.11849715798934,
 ((0, 1, 1), (0, 1, 0, 12)): 449.0583761913741,
 ((1, 0, 0), (0, 1, 0, 12)): 450.19687300224626,
 ((1, 0, 0), (0, 0, 1, 12)): 451.33082188873937,
 ((0, 0, 1), (0, 1, 0, 12)): 451.799176886463,
 ((1, 1, 0), (0, 1, 0, 12)): 458.38926296850997,
 ((0, 1, 0), (0, 1, 0, 12)): 459.7931402248957,
 ((0, 0, 1), (1, 0, 0, 12)): 464.96953508304466,
 ((0, 0, 0), (0, 1, 0, 12)): 474.8232899490605,
 ((0, 0, 0), (1, 0, 0, 12)): 482.51852385919904,
 ((1, 1, 1), (0, 0, 0, 12)): 504.99445549115416,
 ((0, 1, 1), (0, 0, 0, 12)): 516.165922651081,
 ((0, 1, 0), (0, 0, 0, 12)): 519.6567981558159,
 ((1, 1, 0), (0, 0, 0, 12)): 521.6060210726524,
 ((1, 0, 1), (0, 0, 0, 12)): 523.052552949853,
 ((1, 0, 0), (0, 0, 0, 12)): 526.6470324666358,
 ((0, 0, 1), (0, 0, 1, 12)): 602.5706379110426,
 ((0, 0, 0), (0, 0, 1, 12)): 686.1546805384622,
 ((0, 0, 1), (0, 0, 0, 12)): 762.8080147807718,
 ((0, 0, 0), (0, 0, 0, 12)): 869.6198306654262}
Fitting model as per the lowest AIC
# (1, 1, 1), (0, 1, 1, 12)): 351.92946398955104 lowest AIC
​
model = sm.tsa.statespace.SARIMAX(y_train,
                                 order=(1, 1, 1),
                                 seasonal_order=(0, 1, 1, 12),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
​
model_aic = model.fit()
print(model_aic.summary().tables[1])
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.4571      0.150      3.050      0.002       0.163       0.751
ma.L1         -1.0000    566.322     -0.002      0.999   -1110.970    1108.970
ma.S.L12      -0.6781      0.179     -3.787      0.000      -1.029      -0.327
sigma2        22.1252   1.25e+04      0.002      0.999   -2.45e+04    2.46e+04
==============================================================================
Checking MSE & RMSE
forecast = model_aic.get_prediction(start=pd.to_datetime('2022-01-31'), dynamic=False)
predictions = forecast.predicted_mean
​
actual = y_test['2022-01-31':]
mse = ((predictions - actual) ** 2).mean()
rmse = np.sqrt(mse)
​
print('The MSE of our forecasts is {}'.format(round(mse, 2)))
print(f'rmse - : {rmse}')
The MSE of our forecasts is 0.05
rmse - : 0.22275626004126536
Creating SARIMA plot
forecast = model_aic.get_forecast(steps=12)
​
##predictions & confidence interval
predictions=forecast.predicted_mean
ci = forecast.conf_int()
​
​
#Observed Plot
fig = y.plot(label='O3 Actuals', figsize=(14, 7))
fig.set_xlabel('Year', fontsize=12)
fig.set_ylabel('O3 (V µg/m³)', fontsize=12)
fig.fill_between(ci.index,
                ci.iloc[:, 0],
                 ci.iloc[:, 1], color='k', alpha=0.2)
​
#Prediction Plot
predictions.plot(ax=fig, label=' O3 Predictions', alpha= 0.7, figsize=(14,7))
plt.title(' SARIMA Model Showing Predicted O3 Levels in Belfast City Centre 2022', fontsize=20)
plt.legend()
plt.show()
