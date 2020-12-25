# importing necessary packages
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
from scipy.stats import chi2
import numpy as np
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
import math
from statsmodels.tsa.stattools import acf
warnings.filterwarnings("ignore")
# Loading the Seattle dataset
Seattle_data = pd.read_csv("./Historical Hourly Weather Data 2012-2017 - Seattle.csv", header=0, index_col=[0], parse_dates=[0])
print("First five observations from dataset \n", Seattle_data.head())
#Full dataset
print("Full data")
Seattle_data.info()
Seattle_data = Seattle_data.loc['2016-01-01 00:00:00':]
#Dataset description
print("\nData structure - Seattle Dataset")
Seattle_data.info()


#Data Preprocessing
#Proportion of missing values in each columnsS
col = Seattle_data.columns
missing_values = dict()
for i in range(0,len(col)):
    missing_values[col[i]] = (Seattle_data[col[i]].isnull().sum()/len(Seattle_data))*100
missing_values = pd.DataFrame.from_dict(missing_values,orient='index')
print(missing_values)
#Dependent Variable Missing Variable Imputation
#As the missing values are less than 0.0001% - Using Naive method - missing values are imputed with previous value
missing_values_time = Seattle_data[Seattle_data['Temperature'].isnull()].index.to_list()
pos = [Seattle_data.index.get_loc(i) for i in missing_values_time]
for i in pos:
    new_val = Seattle_data['Temperature'].iloc[i-1]
    Seattle_data['Temperature'].iloc[i] = new_val
print("\nAfter interpolation - Number of missing values in dependent variable are")
print(Seattle_data["Temperature"].isnull().sum())

#Independent Variable Missing Variable Imputation
#Humidity
missing_value = Seattle_data['Humidity'].isnull().sum()
while (missing_value != 0):
    missing_values_time = Seattle_data[Seattle_data['Humidity'].isnull()].index.to_list()
    pos = [Seattle_data.index.get_loc(i) for i in missing_values_time]
    for i in pos:
        new_val = Seattle_data['Humidity'].iloc[i - 1]
        if new_val != "":
            Seattle_data['Humidity'].iloc[i] = new_val
            missing_value = missing_value - 1
print("\nAfter interpolation - Number of missing values in independent variable - Humidity are")
print(Seattle_data["Humidity"].isnull().sum())

#Pressure
missing_value = Seattle_data['Pressure'].isnull().sum()
while (missing_value != 0):
    missing_values_time = Seattle_data[Seattle_data['Pressure'].isnull()].index.to_list()
    pos = [Seattle_data.index.get_loc(i) for i in missing_values_time]
    for i in pos:
        new_val = Seattle_data['Pressure'].iloc[i - 1]
        if not np.isnan(new_val):
            Seattle_data['Pressure'].iloc[i] = new_val
            missing_value = missing_value - 1
print("\nAfter interpolation - Number of missing values in independent variable - Humidity are")
print(Seattle_data["Pressure"].isnull().sum())

# Temperature Pattern over the years
# Plot dependent variable versus time
plt.figure()
plt.plot(Seattle_data["Temperature"])
plt.xlabel("Date Time")
plt.ylabel("Temperature")
plt.title("Hourly Weather - Seattle")
plt.show()

# 7 Day pattern over the time
# This helps to understand the seasonal nature of the data in more detail

sample_day = Seattle_data["Temperature"].loc['2016-07-28 00:00:00':'2016-08-05 00:00:00']
y_index = np.arange(0,len(sample_day))
plt.figure()
plt.plot(y_index, sample_day)
plt.xlabel("Date Time")
plt.ylabel("Temperature")
plt.title("Hourly Weather - 7 day pattern")
plt.show()


# **************************************************
# *********** Correlation matrix *******************
# **************************************************
corr_func_sea = Seattle_data[['Humidity', 'Wind Speed', 'Wind Direction', 'Pressure','Temperature']]
correlation = corr_func_sea.corr()
ax = sns.heatmap(correlation, vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200),square=True)
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
ax.set_title("Correlation matrix")
plt.show()

#dependent and independent variable
sea_x = Seattle_data[['Humidity', 'Wind Speed', 'Wind Direction', 'Pressure']]
sea_y = Seattle_data["Temperature"]

# **************************************************
# *********** Seasonal Sub samples *****************
# **************************************************

#Seasonal split
spring_data_y = sea_y.loc['2016-03-01 00:00:00':'2016-06-01 00:00:00']
summer_data_y = sea_y.loc['2016-06-01 00:00:00':'2016-09-01 00:00:00']
fall_data_y = sea_y.loc['2016-09-01 00:00:00':'2016-12-01 00:00:00']
winter_data_y = sea_y.loc['2016-12-01 00:00:00':'2017-03-01 00:00:00']

#Seasonal split
spring_data_x = sea_x.loc['2016-03-01 00:00:00':'2016-06-01 00:00:00']
summer_data_x = sea_x.loc['2016-06-01 00:00:00':'2016-09-01 00:00:00']
fall_data_x = sea_x.loc['2016-09-01 00:00:00':'2016-12-01 00:00:00']
winter_data_x = sea_x.loc['2016-12-01 00:00:00':'2017-03-01 00:00:00']

# **************************************************
# *********** Train and Test Split *****************
# **************************************************
#Linear Regression Train & Test Split
#Adding an Intercept Term
Intercept_term = pd.DataFrame(np.ones((summer_data_x.shape[0],1)), columns=['Intercepts'], index=summer_data_x.index)
summer_data_x = pd.concat([Intercept_term,summer_data_x],axis=1,sort=False)
X_train, X_test, Y_train, Y_test = train_test_split(summer_data_x,summer_data_y,shuffle=False,test_size=0.2)

#Other Model Split
Y_train_spr, Y_test_spr = train_test_split(spring_data_y,shuffle=False,test_size=0.2)
Y_train_sum, Y_test_sum = train_test_split(summer_data_y,shuffle=False,test_size=0.2)
Y_train_fal, Y_test_fal = train_test_split(fall_data_y,shuffle=False,test_size=0.2)
Y_train_win, Y_test_win = train_test_split(winter_data_y,shuffle=False,test_size=0.2)


# **************************************************
# *********** Stationaity Checks *******************
# **************************************************

def ACF_func(sample,lags,label):
    sample = np.array(sample)
    mean_sam = np.mean(sample)
    y_mean = sample - mean_sam
    T = [(np.sum([y_mean[t + 1] * y_mean[t + 1 - k] for t in range(k - 1, len(sample) - 1)]) / (np.sum(y_mean ** 2))) for k in range(0, lags)]
    #print("ACF Values are \n",T)
    #As ACF is symmetric function t[k] = t[-k]
    T_new = np.append(np.flip(T), T[1:])
    plt.figure()
    plt.stem(np.arange(-(lags - 1), lags), T_new, use_line_collection=True)
    plt.xlabel("Lags")
    plt.ylabel("Magnitude")
    plt.title("ACF - {}".format(label))
    plt.show()
    return T
def pcf_plot(x,lags,label):
    sm.graphics.tsa.plot_pacf(x, lags=lags, title="PACF - {}".format(label))
    plt.xlabel("Lags")
    plt.ylabel("Magnitude")
    plt.show()

#ACF & PCF  plots
ACT_t = ACF_func(Y_train_sum,30,"Seattle Hourly Weather Summer Data")
pcf_plot(Y_train_sum,30, "Seattle Hourly Summer Weather Data")

#ADF Test Results
def ADF_test(x):
    test_res = adfuller(x)
    print("ADF Statistics: %f" %test_res[0])
    print("p value: %f" % test_res[1])
    print("Critical Values:")
    for CI,value in test_res[4].items():
        print("\t %s : %0.3f" %(CI,value))
print("\nADF Test Results")
print("\nHourly Weather Analysis - Summer Data")
ADF_test(Y_train_sum)

# *****************************************************
# *********** Seasonal Differencing *******************
# *****************************************************
# 24 differencing
sum_diff24 = Y_train_sum.diff(periods=24)
ACT_t = ACF_func(sum_diff24[24:],60,"Summer Data - seasonal diff 24")
pcf_plot(sum_diff24[24:],60, "Summer Data  - seasonal diff 24")

#24 + 1 differencing
sum_diff_24_1 = sum_diff24[24:].diff()
ACT_t = ACF_func(sum_diff_24_1[1:],60,"Summer Data - seasonal diff + one diff")
pcf_plot(sum_diff_24_1[1:],60, "Summer Data  - seasonal diff + one diff")

print("\nHourly Weather Analysis - Summer Data - Difference Transformed")
ADF_test(sum_diff_24_1[1:])

# For calculating the seasonal components order - ACF & PACF with larger lags is prefered



def STL_decomposition(y,xlab,y_lab):
    #STL_1 = STL(y.iloc[:,0])
    STL_1 = STL(y)
    result = STL_1.fit()
    plt.figure()
    result.plot()
    plt.show()
    T = result.trend
    S = result.seasonal
    R = result.resid
    plt.figure()
    x_ax = np.arange(0,len(y))
    plt.plot(x_ax[:240],T[:240],label="Trend")
    plt.plot(x_ax[:240], S[:240], label="Seasonality")
    plt.plot(x_ax[:240], R[:240], label="residuals")
    plt.title("STL decomposition Plot comparison - Trend,seasonality,Residual")
    plt.xlabel(xlab)
    plt.ylabel(y_lab)
    plt.legend()
    plt.show()
    return T,S,R

T,S,R = STL_decomposition(Y_train_sum,"Series value t","Magnitude")


#Seasonal Adjusted data
Adjusted_seasonal = Y_train_sum - S
x_ax = np.arange(0,len(Y_train_sum))
plt.figure()
plt.plot(x_ax[:240], Y_train_sum[:240], label="Orginal Data")
plt.plot(x_ax[:240], Adjusted_seasonal[:240], label="Seasonal Adjusted")
plt.title("Seasonaly Adjusted plot - STL decomposition")
plt.xlabel("Series t")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

#Detrended data
Adjusted_detrended = Y_train_sum - T
plt.figure()
plt.plot(x_ax[:240], Y_train_sum[:240], label="Orginal Data")
plt.plot(x_ax[:240], Adjusted_detrended[:240], label="detrended")
plt.title("detrended plot - STL decomposition")
plt.xlabel("Series t")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

#Detrended data
detrended_seasonaly_Adj = Y_train_sum - T - S
plt.figure()
#plt.plot(Y_train, label="Orginal Data")
plt.plot(x_ax[:240], detrended_seasonaly_Adj[:240], label="detrended & Seasonaly Adjusted")
plt.title("detrended & Seasonally Adjusted plot - STL decomposition")
plt.xlabel("Series t")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

#Strength of the trend
trend_streng = np.max([0, 1 - (np.var(R)/np.var(T+R))])
print("\n The strength of trend for this data set is ",trend_streng)

#Strength of the Seasonality
seasonal_streng = np.max([0, 1 - ( np.var(R)/np.var(S+R))])
print("\n The strength of seasonality for this data set is",seasonal_streng)



#Helper functions
#Q Value
def ACF(y,error,lag,error_dat):
    ACF_Residuals = ACF_func(error,lag,error_dat)
    q = len(y) * np.sum(np.array(ACF_Residuals[1:]) ** 2)
    return q

#Correlation Coefficient
def correlation_coefficient_cal(X,Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    sq_var_x = []
    sq_var_y = []
    for i in range(0,len(X)):
        sq_var_x.append(X[i]-mean_x)
        sq_var_y.append(Y[i] - mean_y)
    cov_xy = sum([ x*y for x,y in zip(sq_var_x,sq_var_y)])
    cov_x = round(np.sqrt(sum(np.square(sq_var_x))),2)
    cov_y = round(np.sqrt(sum(np.square(sq_var_y))),2)
    r = cov_xy/(cov_x * cov_y)
    return round(r,2)

#HoltsWinter
def holtwinter(train_y, test_y,error,lag,freq,trend,season):
    train_label = train_y.index
    test_label = test_y.index
    train_y_predict = ets.ExponentialSmoothing(train_y,trend=trend, damped=True, seasonal=season, seasonal_periods=freq).fit()
    test_y_predict = train_y_predict.forecast(len(test_y))
    x_val = np.arange(0,len(test_y))
    plt.figure()
    plt.plot(x_val[:720], test_y[:720], label="Test set")
    plt.plot(x_val[:720], test_y_predict[:720], label="Predicted set")
    plt.title("HoltsWinter - h step prediction")
    plt.legend()
    plt.show()
    forecast_error = np.array(test_y) - np.array(test_y_predict)
    predicted_error = np.array(train_y) - np.array(train_y_predict.fittedvalues)
    variance_predicted_error = np.var(predicted_error)
    MSE_predicted_error = np.mean(predicted_error ** 2)
    MSE_forecast_error = np.mean(forecast_error ** 2)
    variance_forecast_error = np.var(forecast_error)
    # Correlation Co efficient
    cc = correlation_coefficient_cal(forecast_error, np.array(test_y))
    # Autocorrelation on prediction error
    if error == "Pred_error":
        error_val = predicted_error
    else:
        error_val = forecast_error
    q = ACF(train_y, error_val, lag, "Holt Winter Residuals")
    stats = [round(MSE_predicted_error,2), round(MSE_forecast_error,2), round(np.mean(predicted_error),2), round(np.mean(forecast_error),2),
             round(variance_predicted_error,2), round(variance_forecast_error,2), round(q,2), cc]
    return stats, test_y_predict

# Average Method
def average_forecast(train_y, test_y,error,lag):
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_y_predict = [np.mean(train_y[:i]) for i in range(1, len(train_y))]
    test_y_predict = [np.mean(train_y[:len(train_y)]) for i in range(len(test_y))]
    predicted_error = np.array(train_y[1:]) - np.array(train_y_predict)
    forecast_error = np.array(test_y) - np.array(test_y_predict)
    MSE_predicted_error = np.mean(predicted_error ** 2)
    MSE_forecast_error = np.mean(forecast_error ** 2)
    variance_predicted_error = np.var(predicted_error)
    variance_forecast_error = np.var(forecast_error)
    cc = correlation_coefficient_cal(forecast_error,test_y)
    #Autocorrelation on prediction error
    if error == "Pred_error":
        error_val = predicted_error
    else:
        error_val = forecast_error
    q = ACF(train_y,error_val,lag,"Average Method Residuals")
    stats = [round(MSE_predicted_error,2), round(MSE_forecast_error,2), round(np.mean(predicted_error),2), round(np.mean(forecast_error),2), round(variance_predicted_error,2), round(variance_forecast_error,2), round(q,2), cc]
    return stats, test_y_predict

# Naive Method
def naive_forecast(train_y, test_y,error,lag):
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_y_predict = train_y[0:len(train_y)-1]
    test_y_predict = [train_y[len(train_y)-1] for i in range(len(test_y))]
    predicted_error = np.array(train_y[1:]) - np.array(train_y_predict)
    forecast_error = np.array(test_y) - np.array(test_y_predict)
    MSE_predicted_error = np.mean(predicted_error ** 2)
    MSE_forecast_error = np.mean(forecast_error ** 2)
    variance_predicted_error = np.var(predicted_error)
    variance_forecast_error = np.var(forecast_error)
    #Correlation Co efficient
    cc = correlation_coefficient_cal(forecast_error, test_y)
    # Autocorrelation on prediction error
    if error == "Pred_error":
        error_val = predicted_error
    else:
        error_val = forecast_error
    q = ACF(train_y,error_val, lag, "Naive Method Residuals")
    stats = [round(MSE_predicted_error,2), round(MSE_forecast_error,2), round(np.mean(predicted_error),2), round(np.mean(forecast_error),2),
             round(variance_predicted_error,2), round(variance_forecast_error,2), round(q,2), cc]
    return stats, test_y_predict

#Drift Method
def drift_forecast(train_y, test_y,error,lag):
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_y_predict = []
    train_y_predict.extend([train_y[0]])
    train_y_predict.extend([train_y[i] + ((train_y[i]-train_y[0])/i) for i in range(1, len(train_y)-1)])
    test_y_predict = [train_y[len(train_y)-1] + (i*((train_y[len(train_y)-1] - train_y[0])/(len(train_y)-1))) for i in range(1, len(test_y)+1)]
    predicted_error = np.array(train_y[1:]) - np.array(train_y_predict)
    forecast_error = np.array(test_y) - np.array(test_y_predict)
    MSE_predicted_error = np.mean(predicted_error ** 2)
    MSE_forecast_error = np.mean(forecast_error ** 2)
    variance_predicted_error = np.var(predicted_error)
    variance_forecast_error = np.var(forecast_error)
    # Correlation Co efficient
    cc = correlation_coefficient_cal(forecast_error, test_y)
    # Autocorrelation on prediction error
    if error == "Pred_error":
        error_val = predicted_error
    else:
        error_val = forecast_error
    q = ACF(train_y,error_val, lag, "Drift Residuals")
    stats = [round(MSE_predicted_error,2), round(MSE_forecast_error,2), round(np.mean(predicted_error),2), round(np.mean(forecast_error),2),
             round(variance_predicted_error,2), round(variance_forecast_error,2), round(q,2), cc]
    return stats, test_y_predict

#SES Method
def ses_forecast(train_y, test_y, a, error,lag):
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_y_predict = []
    train_y_predict.extend([train_y[0]])
    for i in range(0, len(train_y)-1):
        train_y_predict.extend([((a * train_y[i]) + ((1 - a) * train_y_predict[i]))])
    test_y_predict = [ ((a*train_y[len(train_y)-1]) + ((1-a)*(train_y_predict[len(train_y_predict)-1]))) for i in range(0, len(test_y))]
    predicted_error = np.array(train_y) - np.array(train_y_predict)
    forecast_error = np.array(test_y) - np.array(test_y_predict)
    MSE_predicted_error = np.mean(predicted_error[1:] ** 2)
    MSE_forecast_error = np.mean(forecast_error ** 2)
    variance_predicted_error = np.var(predicted_error[1:])
    variance_forecast_error = np.var(forecast_error)
    # Correlation Co efficient
    cc = correlation_coefficient_cal(forecast_error, test_y)
    # Autocorrelation on prediction error
    if error == "Pred_error":
        error_val = predicted_error[1:]
    else:
        error_val = forecast_error
    q = ACF(train_y,error_val, lag, "SES Residuals")
    stats = [round(MSE_predicted_error,2), round(MSE_forecast_error,2), round(np.mean(predicted_error[1:]),2), round(np.mean(forecast_error),2),
             round(variance_predicted_error,2), round(variance_forecast_error,2), round(q,2), cc]
    return stats, test_y_predict

def holtlinear(train_y, test_y, error,lag,freq, trend):
    train_y_predict = ets.ExponentialSmoothing(train_y,trend=trend, damped=True,seasonal=None,seasonal_periods=freq).fit()
    test_y_predict = train_y_predict.forecast(steps=len(test_y))
    forecast_error = np.array(test_y) - np.array(test_y_predict)
    predicted_error = np.array(train_y) - np.array(train_y_predict.fittedvalues)
    variance_predicted_error = np.var(predicted_error)
    MSE_predicted_error = np.mean(predicted_error ** 2)
    MSE_forecast_error = np.mean(forecast_error ** 2)
    variance_forecast_error = np.var(forecast_error)
    x_val = np.arange(0, len(test_y))
    plt.figure()
    plt.plot(x_val[:720], test_y[:720], label="Test set")
    plt.plot(x_val[:720], test_y_predict[:720], label="Predicted set")
    plt.title("HoltsLinear - h step prediction")
    plt.legend()
    plt.show()
    # Correlation Co efficient
    cc = correlation_coefficient_cal(forecast_error, np.array(test_y))
    # Autocorrelation on prediction error
    if error == "Pred_error":
        error_val = predicted_error
    else:
        error_val = forecast_error
    q = ACF(train_y, error_val, lag, "Holt Linear Residuals")
    stats = [round(MSE_predicted_error,2), round(MSE_forecast_error,2), round(np.mean(predicted_error),2), round(np.mean(forecast_error),2),
             round(variance_predicted_error,2), round(variance_forecast_error,2), round(q,2), cc]
    return stats, test_y_predict

#Plotting Forecast Model - Function
def sub_plt(rows, cols, title, sub_title, y_train, y_test, forecast_val, x_label, y_label, freq):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, figsize=(15,7))
    fig.subplots_adjust(hspace=0.8, wspace=0.5, top=0.85)
    fig.suptitle(title)
    for ax, y, sub_tit in zip(axes.flatten(), forecast_val, sub_title):
        ax.plot(y_train.index, y_train, label="Training Set")
        ax.plot(y_test.index, y_test, color='orange', label="Testing Set")
        ax.plot(y_test.index, y, color='green', label="h-step forecast")
        ax.set(title=sub_tit, xlabel=x_label, ylabel=y_label)
        ax.xaxis.label.set_size(10)
        ax.title.set_size(10)
        ax.legend(fontsize=10)
        fig.autofmt_xdate()
        if freq != 1:
            ax.get_xticklabels()
            ax.set_xticks(ax.get_xticks()[::freq])
    fig.tight_layout()
    plt.show()

#Base Model Evaluation
Forecast_Model_Sea = pd.DataFrame(columns=['Average Forecast', 'Naive', 'Drift', 'SES-alpha = 0.5','Holts_Linear','Holts_winter'])
Forecast_Model_Sea['Average Forecast'], forecast_val_avg = average_forecast(Y_train_sum, Y_test_sum, "Pred_error",30)
Forecast_Model_Sea['Naive'], forecast_val_naive = naive_forecast(Y_train_sum, Y_test_sum, "Pred_error",30)
Forecast_Model_Sea['Drift'],forecast_val_drift = drift_forecast(Y_train_sum, Y_test_sum, "Pred_error",30)
Forecast_Model_Sea['SES-alpha = 0.5'], forecast_val_ses = ses_forecast(Y_train_sum, Y_test_sum, 0.5, "Pred_error",30)
Forecast_Model_Sea['Holts_Linear'],forecast_val_holtlin = holtlinear(Y_train_sum, Y_test_sum, "Pred_error",30,24,'add')
Forecast_Model_Sea['Holts_winter'],forecast_val_holtwin = holtwinter(Y_train_sum, Y_test_sum, "Pred_error",30,24,'add','mul')
Forecast_Model_Sea = Forecast_Model_Sea.set_index(pd.Series(['MSE_pred','MSE_Forecast','Mean_pred','Mean_Forecast','Variance_pred','Variance_Forecast','Q Value', 'Correlation coefficient']))

forecast_val = [list(forecast_val_avg), list(forecast_val_naive), list(forecast_val_drift), list(forecast_val_ses), list(forecast_val_holtlin), list(forecast_val_holtwin)]
sub_title = ['Average Forecast Model', 'Naive Forecast Model', 'Drift Forecast Model', 'Ses-alpha=0.5 Forecast Model', 'Holt Linear Forecast Model', 'Holt Winter Forecast Model']

sub_plt(3,2,"Forecast Model Result", sub_title, Y_train_sum, Y_test_sum, forecast_val, 'Date(Year)', 'Temperature', 1 )

print(Forecast_Model_Sea[['Average Forecast', 'Naive', 'Drift', 'SES-alpha = 0.5']])
print(Forecast_Model_Sea[['Holts_Linear','Holts_winter']])



#Linear Regression Train & Test Split
#Adding an Intercept Term
Intercept_term = pd.DataFrame(np.ones((sea_x.shape[0],1)), columns=['Intercepts'], index=sea_x.index)
sea_x = pd.concat([Intercept_term,sea_x],axis=1,sort=False)
X_train, X_test, Y_train, Y_test = train_test_split(sea_x,sea_y,shuffle=False,test_size=0.2)

# **************************************************************
# *********** Linear Regression ********************************
# **************************************************************
model = sm.OLS(Y_train,X_train).fit()
print("\nModel Summary")
print(model.summary())

# ***************************************************************************
# *********** Linear Regression Prediction & Plot ***************************
# ***************************************************************************
predicts = model.predict(X_test)
plt.figure()
plt.title("Linear Regression Weather Dataset - Prediction")
plt.plot(Y_train, label="Train_Dataset")
plt.plot(Y_test, label="test")
plt.plot(predicts,label="Predictions")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

# ***************************************************************************
# ************* Forecast Errors & ACF ***************************************
# ***************************************************************************
forecast_error = Y_test - predicts
ACT_t = ACF_func(forecast_error, 30, "Linear Regression Forecast errors")

af = correlation_coefficient_cal(Y_test,predicts)

# ***************************************************************************
# ************* Estimated Variance - Forecast Errors ************************
# ***************************************************************************
T = X_test.shape[0]
K = X_test.shape[1]-1
const = (1 / (T-K-1))
SSE = np.sum(np.square(forecast_error))
variance_forec = const*SSE
SD_forec = np.sqrt(variance_forec)
print("Variance of the Forecast error is given as",variance_forec)
print("SD of the Forecast error is given as",SD_forec)

# ***************************************************************************
# ************* Predicted Errors & ACF ***************************************
# ***************************************************************************
pred_values = model.predict(X_train)
predict_error = Y_train - pred_values
ACT_t = ACF_func(predict_error, 30, "Linear Regression Residuals")

a = correlation_coefficient_cal(Y_train,pred_values)
print("\ncorrelation coefficients predicted value and original set:", a)

# ***************************************************************************
# ************* Estimated Variance - Residuals ************************
# ***************************************************************************
T = X_train.shape[0]
K = X_train.shape[1]-1
const = (1 / (T-K-1))
SSE = np.sum(np.square(predict_error))
variance_pred = const*SSE
SD_pred = np.sqrt(variance_pred)
print("Variance of the Residuals is given as",variance_pred)
print("SD of the Residuals is given as",SD_pred)

# ***************************************************************************
# ************* Q Value - Residuals ************************
# ***************************************************************************

q = ACF(X_train, predict_error, 30, "Linear Regression Residuals")
print("Q value of Linear Regression Model Residuals: ", q)

# ***************************************************************************
# *********** Linear Regression Prediction & Plot ***************************
# ***************************************************************************
predicts = model.predict(X_test)
x_ind = np.arange(0,len(Y_test))
plt.figure()
plt.title("Linear Regression Weather Dataset - Prediction")
plt.plot(x_ind[:720],Y_test[:720], label="test")
plt.plot(x_ind[:720],predicts[:720],label="Predictions")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.legend()
plt.show()


print("\nLinear Regression Results")
print("Mean of the Residuals: ", np.mean(predict_error))
print("Variance of the Residuals: ",variance_pred)
print("MSE Residuals: ", np.square(np.mean(predict_error)))
print("Q Value: ", q)
print("Mean of the Forecast Error: ", np.mean(forecast_error))
print("Variance of the Forecast Error: ",variance_forec)
print("MSE Forecast Error: ", np.square(np.mean(forecast_error)))
print("Correlation coefficients predicted value and test set:", af)
print("correlation coefficients predicted value and original set:", a)



#Helper functions
#Gpac Function
def GPAC(ry,a,b):
    gpac_mat = []
    for j in range(0,b):
        row = []
        for k in range(1,a+1):
            mat_ele = [[ry[np.abs(n)] for n in range(j-m,k+j-m)] for m in range(0,k-1)]
            last_num = [ry[np.abs(n)] for n in range(j+1,k+j+1)]
            last_den = [ry[np.abs(n)] for n in range(j-k+1,j+1)]
            num = mat_ele.copy()
            den = mat_ele.copy()
            num.append(last_num)
            den.append(last_den)
            num = np.array(num).transpose()
            den = np.array(den).transpose()
            #gpac = np.round((np.linalg.det(num) / np.linalg.det(den)),3)
            det_num = np.linalg.det(num)
            det_den = np.linalg.det(den)
            if det_den == 0.0:
                gpac = math.inf
            else:
                gpac = det_num/det_den
            row.append(gpac)
        gpac_mat.append(row)
    GPAC_tab = pd.DataFrame(gpac_mat, columns=list(np.arange(1, a+1)))
    return GPAC_tab

#mean subtraction
y = sum_diff_24_1[1:] - np.mean(sum_diff_24_1[1:])

#Find ry - autocorrelation values with lags of 50
ry = ACF_func(y, 60,'Seattle Weather data after detrended and seasonaility adj')
G_pac = GPAC(ry,8,8)

print("\nGPAC Results\n")
print(G_pac)
G_pac.replace(np.inf, np.nan, inplace=True)
# Heat map
ax = sns.heatmap(G_pac, center=0, cmap=sns.color_palette('rocket_r'), annot=True, linewidths=.5)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels())
plt.title("Generalized Partial Autocorrelation(GPAC) table")
plt.show()

#PACF Plots
pcf_plot(sum_diff_24_1[1:],60, "Summer Data  - seasonal diff + one diff")

# Slicing 24 lagged component values
acf_24 = acf(y,nlags = 216)
pacf_24 = pacf(y,nlags = 216)
acf_seas = acf_24[::24]
pacf_seas = pacf_24[::24]

#plotting
lag_com = np.arange(24,240,24)
x = [0]
x.extend(lag_com)
plt.figure()
plt.stem(x,acf_seas)
plt.title("ACF - 24 lagged component")
plt.xticks(x)
plt.show()

plt.figure()
plt.stem(x,pacf_seas)
plt.title("PACF - 24 lagged component")
plt.xticks(x)
plt.show()


#Parameter Estimation
#Helper Functions
#Step 1 - Lavenberg Marquardf Algorithm
def gardient_Cal(y,theta,na,nb,sigma):
    den = [1]
    den.extend(theta[:na])
    num = [1]
    num.extend(theta[na:])
    if len(num) != len(den):
        diff = np.abs(len(num) - len(den))
        diff = list(np.zeros(diff, dtype=int))
        if len(num) < len(den):
            num.extend(diff)
        else:
            den.extend(diff)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    n = na+nb
    X = []
    for i in range(0,na):
        den_new = den.copy()
        den_new[i+1] = den[i+1]+sigma
        num_new = num
        sys = (den_new, num_new, 1)
        _, e_theta = signal.dlsim(sys, y)
        x = (e - e_theta)/sigma
        x = x.flatten()
        X.append(x.transpose())
    for i in range(0,n-na):
        num_new = num.copy()
        num_new[i+1] = num[i+1]+sigma
        den_new = den
        sys = (den_new, num_new, 1)
        _, e_theta = signal.dlsim(sys, y)
        x = (e - e_theta)/sigma
        x = x.flatten()
        X.append(x.transpose())
    X = np.array(X).transpose()
    A = np.matmul(X.transpose(), X)
    g = np.matmul(X.transpose(),e)
    SSE = np.matmul(e.transpose(), e)
    return A,SSE,g

#Step 2 - Lavenberg Marquardf Algorithm
def max_prob_lm_alg(A,g,y,u,na,nb, theta):
    n = na+nb
    I = np.identity(n)
    det_theta = np.matmul(np.linalg.inv((A + (u*I))),g)
    det_theta = det_theta.flatten()
    theta_new = theta+det_theta
    den = [1]
    den.extend(theta_new[:na])
    num = [1]
    num.extend(theta_new[na:])
    if len(num) != len(den):
        diff = np.abs(len(num) - len(den))
        diff = list(np.zeros(diff, dtype=int))
        if len(num) < len(den):
            num.extend(diff)
        else:
            den.extend(diff)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    SSE_new = np.matmul(e.transpose(), e)
    return SSE_new,theta_new,det_theta

#Step 3 - Lavenberg Marquardf Algorithm
def lm_alg(iter_max,SSE_new,SSE,det_theta,theta_new,na,nb,A,u,u_max,y,sigma,g):
    N = len(y)
    n = na+nb
    SSE_plot = [SSE[0][0]]
    for i in range(1,iter_max):
        SSE_plot.append(SSE_new[0][0])
        if SSE_new < SSE:
            if np.linalg.norm(det_theta,2) < (10**-4):
                theta_hat = theta_new
                sigma_sq = (SSE_new)/(N-n)
                cov = sigma_sq * np.linalg.inv(A)
                print("Max Iteration for Convergence: ", i)
                return theta_hat, cov, SSE_plot
            else:
                theta=theta_new
                u=u/10
                A,SSE,g = gardient_Cal(y, theta, na, nb, sigma)
                SSE_new, theta_new, det_theta = max_prob_lm_alg(A, g,y, u, na, nb, theta)
        while(SSE_new >= SSE):
            u = u*10
            if u > u_max:
                print("u is greater than max of u - 10^10")
                return 0,0,0
            SSE_new, theta_new, det_theta = max_prob_lm_alg(A,g, y,u,na,nb,theta)
        if i == iter_max-1:
            print("Maximum Iteration reached")
            return 0, 0,0
#Confidence Interval
def confidence_interval(theta,covar,na,nb):
    upper_interval = [theta[i]+(2*np.sqrt(covar[i][i])) for i in range(0,len(theta))]
    lower_interval = [theta[i] - (2 * np.sqrt(covar[i][i])) for i in range(0, len(theta))]
    print("\nConfidence Interval")
    for i in range(0,na):
        print("{} < a{} < {}".format(lower_interval[i],i+1,upper_interval[i]))
    for i in range(0,nb):
        print("{} < b{} < {}".format(lower_interval[i+na],i+1,upper_interval[i+na]))

#One Step Prediction
#ALl the initial conditions are assumed as zero
def one_step_predictions(train,y_para,e_para):
    predicts = [0]
    train = list(train)
    y = np.array(train).reshape(len(train),)
    na_len = len(y_para)
    nb_len = len(e_para)
    #Y parameter values
    pred_y_term = [np.sum([y[i-j-1]*(-y_para[j]) for j in range(0, na_len) if i-j > 0]) for i in range(1, len(y))]
    if nb_len !=0:
        pred_y_term[0] = pred_y_term[0] + (e_para[0]*y[0])
        predicts.append(pred_y_term[0])
        prediction_error = list(y[:2] - predicts)
        #e coeeficients
        for i in range(2,len(y)):
            pred_e_terms = np.sum([((y[i-j-1] * e_para[j]) - predicts[i-j-1]*(e_para[j])) for j in range(0, nb_len) if i-j >= 1])
            predicts.append(pred_y_term[i-1]+pred_e_terms)
            preds = y[i] - predicts[i]
            prediction_error.append(preds)
    else:
        predicts.extend(pred_y_term)
        prediction_error = list(y - predicts)
    return predicts,prediction_error

#Correlation co-efficient
def correlation_coefficient_cal(X,Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    sq_var_x = []
    sq_var_y = []
    for i in range(0,len(X)):
        sq_var_x.append(X[i]-mean_x)
        sq_var_y.append(Y[i] - mean_y)
    cov_xy = sum([ x*y for x,y in zip(sq_var_x,sq_var_y)])
    cov_x = round(np.sqrt(sum(np.square(sq_var_x))),2)
    cov_y = round(np.sqrt(sum(np.square(sq_var_y))),2)
    r = cov_xy/(cov_x * cov_y)
    return round(r,2)

#From the GPAC Results
#ARMA (2,0) Process
na = 2
nb = 0


#Initializing the Paramters for the LM

sigma = 10**-6
iter_max = 100
u_max = 10**10
u = 0.01

#Parameter Estimation
# Step 0 - LM Algorithm
theta = list(np.zeros(na + nb, dtype=int))
#Step 1 - LM Algorithm
A, SSE_old, g = gardient_Cal(y, theta, na, nb, sigma)
#Step 2- LM ALgorithm
SSE_new, theta_new, det_theta = max_prob_lm_alg(A, g, y, u, na, nb, theta)
print("\nLM Algorithm Results")
#Step 3 - LM Algorthm
theta_hat, cov, SSE_vals = lm_alg(iter_max, SSE_new, SSE_old, det_theta, theta_new, na, nb, A, u, u_max, y, sigma, g)
if np.all(theta_hat) == True:
    print("\nFinal Parameters")
    print(theta_hat)
    confidence_interval(theta_hat, cov, na, nb)
    #One step prediction
    y_para = theta_hat[:na]
    e_para = theta_hat[na:]
    predict_val, pred_error = one_step_predictions(y, y_para, e_para)
    y_ax = np.arange(1, len(y) + 1)
    plt.figure()
    plt.plot(y_ax[1:500], y[1:500], label=["Training Set"])
    plt.plot(predict_val[1:500], label=["One-step Predicted Set"])
    plt.xlabel("TimeSteps")
    plt.ylabel("Y Values")
    plt.legend()
    plt.title("ARMA ({},{}) Process - One Step Prediction".format(na,nb))
    plt.show()
    #Residual Errors
    ACF_Residuals = ACF_func(pred_error, 30, "ARMA ({},{}) Residuals".format(na,nb))
    mean_error = np.mean(pred_error)
    var_error = np.var(pred_error)
    MSE_predicted_error = np.mean(np.array(pred_error) ** 2)
    q = len(y) * np.sum(np.array(ACF_Residuals[1:]) ** 2)
    print("\nQ Value of the Residuals: ", q)
    print("\nChi-Square Whiteness Test")
    DOF = 30 - na - nb
    alfa = 0.05
    chi_critical = chi2.ppf(1 - alfa, DOF)
    print("Q critical value from chi2 table - ", chi_critical)
    if q < chi_critical:
        print("The Residuals are White")
    else:
        print("Residuals are not White")
    #Residual Error stats
    mean_error = np.mean(pred_error)
    var_error = np.var(pred_error)
    MSE_predicted_error = np.mean(np.array(pred_error) ** 2)
    print("\nResidual error stats")
    print("Mean of the Residual error", mean_error)
    print("Variance of the residual error", var_error)
    print("MSE of the error", MSE_predicted_error)
    num_val = [1]
    num_val.extend(y_para)
    den_val = [1]
    den_val.extend(e_para)
    num_roots = np.roots(num_val)
    den_roots = np.roots(den_val)
    print("The Roots of the numerators are ", num_roots)
    print("The Roots of the denominators are ", den_roots)
    print("The covariance matrix for the estimated parameters : ", cov)
    # PACF Plots
    pcf_plot(pred_error, 60, "ARMA({},{}) Residuals".format(na,nb))

#ARMA Model from the package to check the co-efficient results
from statsmodels.tsa.arima_model import ARIMA

# 1
model = ARIMA(y, order=(na,0,nb))
model_fit = model.fit(disp=0)
print(model_fit.summary())

ARMA_Model_Sea = pd.DataFrame(columns=['ARMA (2,0)'])
ARMA_Model_Sea['ARMA (2,0)'] = [MSE_predicted_error, mean_error, var_error, q,"No" ]
ARMA_Model_Sea = ARMA_Model_Sea.set_index(pd.Series(['MSE_pred','Mean_pred','Variance_pred','Q Value', 'Chi square test passed']))

#ALl initial conditions are set to zero
def multi_step_predict(train,test,y_para,e_para):
    predicts = []
    train = np.array(train).reshape(len(train),)
    test = np.array(test).reshape(len(test),)
    train_len = len(train)
    na_len = len(y_para)
    nb_len = len(e_para)
    if nb_len != 0:
        if na_len != 0:
            for i in range(0,na_len):
                y_term = np.sum([-train[train_len-j+i]*y_para[j-1] for j in range(1, na_len+1) if i-j < 0])
                y_term_fut = np.sum([-predicts[i-j]*y_para[j-1] for j in range(1, na_len+1) if i-j >= 0])
                y_term = y_term+y_term_fut
                if i < nb_len:
                    e_term = np.sum([train[train_len-j+i]*e_para[j-1] for j in range(1, nb_len+1) if i-j < 0])
                else:
                    e_term = 0.0
                preds = y_term+e_term
                predicts.append(preds)
            for i in range(na_len, len(test)):
                y_term = np.sum([-predicts[i - j] * y_para[j - 1] for j in range(1, na_len + 1)])
                if nb_len <= i:
                    predicts.append(y_term)
                else:
                    e_term = np.sum([train[train_len - j] * e_para[j - 1] for j in range(1, nb_len + 1)])
                    preds = y_term + e_term
                    predicts.append(preds)
        else:
            for i in range(0,nb_len):
                e_term = np.sum([train[train_len - j + i] * e_para[j - 1] for j in range(1, nb_len + 1) if i - j < 0])
                predicts.append(e_term)
            for i in range(nb_len,len(test)):
                e_t = 0
                predicts.append(e_t)
    else:
        for i in range(0,na_len):
            y_term = np.sum([-train[train_len-j+i]*y_para[j-1] for j in range(1, na_len+1) if i-j < 0])
            y_term_fut = np.sum([-predicts[i-j]*y_para[j-1] for j in range(1, na_len+1) if i-j >= 0])
            y_term = y_term+y_term_fut
            predicts.append(y_term)
        for i in range(na_len, len(test)):
            y_term = np.sum([-predicts[i - j] * y_para[j - 1] for j in range(1, na_len + 1)])
            predicts.append(y_term)
    return predicts



#ARMA (3,0) Model

na = 3
nb = 0

#Parameter Estimation
# Step 0 - LM Algorithm
theta = list(np.zeros(na + nb, dtype=int))
#Step 1 - LM Algorithm
A, SSE_old, g = gardient_Cal(y, theta, na, nb, sigma)
#Step 2- LM ALgorithm
SSE_new, theta_new, det_theta = max_prob_lm_alg(A, g, y, u, na, nb, theta)
print("\nLM Algorithm Results")
#Step 3 - LM Algorthm
theta_hat, cov, SSE_vals = lm_alg(iter_max, SSE_new, SSE_old, det_theta, theta_new, na, nb, A, u, u_max, y, sigma, g)
if np.all(theta_hat) == True:
    print("\nFinal Parameters")
    print(theta_hat)
    confidence_interval(theta_hat, cov, na, nb)
    #One step prediction
    y_para = theta_hat[:na]
    e_para = theta_hat[na:]
    predict_val, pred_error = one_step_predictions(y, y_para, e_para)
    y_ax = np.arange(1, len(y) + 1)
    plt.figure()
    plt.plot(y_ax[1:500], y[1:500], label=["Training Set"])
    plt.plot(predict_val[1:500], label=["One-step Predicted Set"])
    plt.xlabel("TimeSteps")
    plt.ylabel("Y Values")
    plt.legend()
    plt.title("ARMA ({},{}) Process - One Step Prediction".format(na,nb))
    plt.show()
    #Residual Errors
    ACF_Residuals = ACF_func(pred_error, 30, "ARMA ({},{}) Residuals".format(na,nb))
    mean_error = np.mean(pred_error)
    var_error = np.var(pred_error)
    MSE_predicted_error = np.mean(np.array(pred_error) ** 2)
    q = len(y) * np.sum(np.array(ACF_Residuals[1:]) ** 2)
    print("\nQ Value of the Residuals: ", q)
    print("\nChi-Square Whiteness Test")
    DOF = 30 - na - nb
    alfa = 0.05
    chi_critical = chi2.ppf(1 - alfa, DOF)
    print("Q critical value from chi2 table - ", chi_critical)
    if q < chi_critical:
        print("The Residuals are White")
    else:
        print("Residuals are not White")
    #Residual Error stats
    mean_error = np.mean(pred_error)
    var_error = np.var(pred_error)
    MSE_predicted_error = np.mean(np.array(pred_error) ** 2)
    print("\nResidual error stats")
    print("Mean of the Residual error", mean_error)
    print("Variance of the residual error", var_error)
    print("MSE of the error", MSE_predicted_error)
    num_val = [1]
    num_val.extend(y_para)
    den_val = [1]
    den_val.extend(e_para)
    num_roots = np.roots(num_val)
    den_roots = np.roots(den_val)
    print("The Roots of the numerators are ", num_roots)
    print("The Roots of the denominators are ", den_roots)
    print("The covariance matrix for the estimated parameters : ", cov)
    # PACF Plots
    pcf_plot(pred_error, 60, "ARMA({},{}) Residuals".format(na,nb))

#ARMA Model from the package to check the co-efficient results
from statsmodels.tsa.arima_model import ARIMA

# 1
model = ARIMA(y, order=(na,0,nb))
model_fit = model.fit(disp=0)
print(model_fit.summary())


ARMA_Model_Sea['ARMA (3,0)'] = [MSE_predicted_error, mean_error, var_error, q,"No" ]


#SARIMAX - Input is direct value without differencing values as package takes care of differencing part

smodel = sm.tsa.statespace.SARIMAX(Y_train_sum, order=(2,1,0), seasonal_order=(0,1,1,24)).fit()
print(smodel.summary())

preds = smodel.fittedvalues
pred_error = np.array(Y_train_sum) - np.array(preds)

ACF_Residuals = ACF_func(pred_error, 30, "SARiMA (2,1,0)(0,1,1,24) Residuals")

mean_error = np.mean(pred_error)
var_error = np.var(pred_error)
MSE_predicted_error = np.mean(np.array(pred_error) ** 2)
print("\nResidual error stats")
print("Mean of the Residual error", mean_error)
print("Variance of the residual error", var_error)
print("MSE of the error", MSE_predicted_error)
y_ax = np.arange(1,len(Y_train_sum)+1)
plt.figure()
plt.plot(np.array(y_ax), np.array(Y_train_sum), label=["Training Set"])
plt.plot(np.array(y_ax), np.array(preds), label=["One-step Predicted Set"])
plt.xlabel("TimeSteps")
plt.ylabel("Y Values")
plt.legend()
plt.title("SARIMA (2,1,0)(2,1,1,24) Process - One Step Prediction")
plt.show()
#Q value caluclation
q = len(y) * np.sum(np.array(ACF_Residuals[1:]) ** 2)
print("\nQ Value of the Residuals: ", q)
print("\nChi-Square Whiteness Test")
DOF = 30 - 2 - 0 - 2 - 1
alfa = 0.05
chi_critical = chi2.ppf(1 - alfa, DOF)
print("Q critical value from chi2 table - ", chi_critical)
if q < chi_critical:
    print("The Residuals are White")
else:
    print("Residuals are not White")


ARMA_Model_Sea['SARIMA (2,1,0),(0,1,1)'] = [MSE_predicted_error, mean_error, var_error, q,"No" ]

print("Model Performances on the ARMA Models")
print(ARMA_Model_Sea)







