# Weather Forecasting - Hourly Temperature of Seattle City

Weather forecast is the branch of science to predict the conditions of the atmosphere for the given location and time. This is more relatable as this helps to plan everyday travel and other related activities. Weather warnings are the most important forecasts as they protect life and property from adverse damage. In this project, I have used hourly temperature data of the Seattle city and built a prediction model to forecast the upcoming temperature. I have made use of various time series model techniques like average method, naïve method, drift method, simple exponential smoothing, holts’ linear method, holts winter method and ARMA methods to build the prediction model. Also, I have performed multivariate regression analysis on the dataset to check the linear dependency of the target variable with the regressor. A comparative study is performed to determine the best model by evaluating the results of these models like MSE values, variance & mean of the predicted error & forecast error, Q value and chi square test results. Using this best model,the h step forecast is performed on the test set.

## Temperature Pattern Over time

![Temp](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/temp.JPG)

From the above plot, we could observe strong trend, seasonality, and cyclic pattern present in the dataset.

Let’s take a closer look at the seasonal part of the dataset, to understand more on the hourly data pattern. For this, I have sampled 7-day data from the dataset and plotted the data. 

![7day](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/7%20day%20pattern.JPG)

This plot uses the data from July 28, 2016 00:00:00 to Aug 04,2016 23:00:00. We could see a repeating pattern every 24-time cycle. Also, seasonal spike is not the same, this information is helpful while selecting “add” or “mul” decomposition values in holts’ winter/linear method.
Due to this cyclic nature and multiple seasonality, there is possibility that data may be highly nonlinear. To overcome this issue, I have resampled the data as per seasonal order and developed four models. Hence, data is sub-sampled as Spring data – March to May, Summer data – June to August, Fall data – September to November and Winter data – December to Feb. For this split, I have used 2016 data only. In this project, I have used Summer data to built the model and run the prediction over it. 

## Stationarity Test

![test1](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/stationarity_test1.JPG)

Although ADF test result suggests data is stationary with p value less than significance value of 0.05 and confidence interval of 95% as ASDF stats is less than 5% of the CI value, ACF plot lags values are decaying slowly with repeating pattern of 24 lags suggests that data is not stationary. This calls for the transformation like differencing. 
Due to the seasonal nature of the data, I have performed seasonal differencing of period 24 over the data set. Let’s check the ACF plot for this differenced data

![test2](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/stationarity_test2.JPG)

From the above plots, we could see that seasonal difference transformation has adjusted the repeating pattern to the maximum extent. However, we could still see the ACF is decaying slowly, hence I have used normal differential transformation on the data. 

![test3](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/stationarity_test3.JPG)

From the above plots and ADF tests we could see data became stationary with p value less then the significance value plus the ADF stats is far lower than 1% CI value suggesting more than 99% confidence interval. 

## Time Series Decomposition

I have used the additive STL decomposition to approximate trend, seasonality from the original dataset. 

![timedecomp](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/time_Series_decompo.JPG)

From the plots, we could infer that STL decomposition works well for our data and able to capture most of the trends and seasonality present in our data. Variability present in the dataset is captured and removed from the data. Looks like both trend and seasonal components dominate our data, we will confirm the same as below

![strength](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/stren.JPG)

We could see data has both trend and seasonality to the maximum, we could say there are higher chances that data might be nonlinear.

## Conventional Basic Model

We have applied basic models like average, naïve, drift, SES, holts linear and holts winter method to our train set and made an h step prediction over the test data. All the basic stats values like MSE, Mean, Variance and Q value is calculated on the prediction error & forecast errors to do the comparative analysis over the model. 

![Base](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/conventional_model.JPG)

ACF plots also reveal that the holt’s winter one step prediction is almost equal to the impulse response (i.e.) white noise. Next closest model is holt’s linear model. Let’s give a closer look to the h step prediction of these models to understand the pattern. 

![ACF](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/conve_acf.JPG)

![Results](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/conve_res.JPG)

Among base models, MSE of the holt’s winter method is lowest and the mean of the prediction error is almost 0 and variance is 0.37. Although it could not be able to capture exact variability, there exists some correlation between the predicted values and actual values. With Q value far less than other models, suggests holts winter method outperforms other basic conventional forecast models. 

## ARMA Models

### Order Estimation

The potential order for the ARMA model can be calculated from the autocorrelation lag behavior present in the data. This checks possible correlation between the values to find best possible correlation value between the y(t) and y(t-h). let’s calculate ACF, PACF and GPAC to find the possible order for our data. 

![GPAC](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/gpac.JPG)

![Lag](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/lagged_com.JPG)

From the GPAC, we could see the potential order to be as follows
1. na = 2, nb = 0
2. na = 3, nb = 0
3. na = 4, nb = 1
4. na = 6, nb = 5

PACF plot – two lags have significant value above the blue line – suggesting AR(2) to the model, while ACF plot -> lag value follows the tailing off pattern leaving MA(0) model. Thus, from the ACF & PACF plot – the potential pattern is na = 2 & nb = 0. We were able to see a similar pattern from the GPAC table as well. Let’s calculate the estimate for these potential orders and perform ARMA model. 

## ARMA(2,0) Model

### Paramter Estimation

![ARMA_1](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/arma_1_co.JPG)

Since Stats model package brings the co-efficient to the right side, the negative sign is neglected. We could see both the results match.

ARMA (2,0) Model is given as

![ARMA_model](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/arma_model.JPG)

### One step Prediction & Residuals

![ARMA_model_plots](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/arma1_pl.JPG)

From the above one step prediction stats, we could see the mean and variance is not almost equal to the (0,1), also the Q value is huge. The plots suggest that the model couldn’t be predicted properly. Also, the residual plot is not close to white noise as we could see a sharp spike at the interval k = 24,48, etc. 

### Chi Square Test

From the residual plot, we could say that this model isn’t able to capture the entire information present in the data, as residuals do contain some information that are reflected by the large spike in the ACF. And the Q value is also high around 490. Let’s compare this value with q_critical and confirm the chi square test results on residual pattern. 

![chi square 1](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/chi_Sq1.JPG)

Chi square test results suggest that the residual errors are not white. Thus, chi square test failed for this model. This model is not the significant model.

Let’s try the same approach for other estimated orders.

## ARMA(3,0) Model

![Arma2](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/Arma2.JPG)

Since Stats model package brings the co-efficient to the right side, the negative sign is neglected. We could see both the results match.

ARMA (3,0) Model is given as

![Arma2_model](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/arma2_model.JPG)

### One step Prediction & Residuals

![Arma2_model](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/Arma2_plo.JPG)

From the above one step prediction stats, we could see the mean and variance is not almost equal to the (0,1), also the Q value is huge. The plots suggest that model couldn’t be
predicted properly. Also, the residual plot is not close to white noise as we could see a sharp spike at the interval k = 24,48, etc. 

### Chi Square Test

From the residual plot, we could say that this model isn’t able to capture the entire information present in the data, as residuals do contain some information that are reflected by the large spike in the ACF. And the Q value is also high around 491. Let’s compare this value with q_critical and confirm the chi square test results on the residual pattern. 

![Arma2_model_chi](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/chi_sq2.JPG)

Chi square test results suggest that the residual errors are not white. Thus, chi square test failed for this model. This model is not the significant model. 

I have also tried other na,nb values and chi square tests but failed for those models as well. One of the possible reasons for the chi square test failure may be non-linearity present in our data. Also, we have calculated the strength of the trend and seasonality in our dataset which is greater than 90%. Thus, remaining residuals dont constitute much to the data series. From the ACF of the residual values, we could see a sharp spike at lags k =24,48, this suggests there is more information left in the seasonal components of the dataset at the interval of 24. Thus, normal ARMA models do not produce good results with our dataset. 

As our data have high seasonality, I planned to experiment with the SARIMA model instead of ARIMA. Since ARIMA model expects the data to be non-seasonal, our earlier research suggests us that more information may be available at seasonal components, so I have skipped the ARIMA part and moved to the the SARIMA. 

## SARIMA - Seasonal ARIMA

The SARIMA model takes care of both seasonality and trend present in the data. The input to the model contains both seasonal components and non-seasonal components. And, the
integration value in the components takes care of the trend present in the data. One major challenge with SARIMA is to find the order for the seasonal component. 

From the earlier analysis, we know that potential non-seasonal components might be ( 2,0) , (3,0),(4,1), (6,5). From the lagged ACF plot suggests that ACF is having a sharp spike and cutsoff after that, this gives a MA(1) model while PACF decays through the lags – this constitutes to the AR (0). Hence, potential order for the seasonal components would be (0,1). 

Lets built SARIMA model with these values and check for residuals and chi square test values for more information. 

Possible model – ARIMA (2,1,0) (0,1,1,24)

![SARIMA](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/sarima.JPG)

From the SARIMA outcome, we could see that Q value is dropped to 290 and all the p values of the coefficients are significant. But the prob of Q values is 0.00 which rejects the chi square test. The one potential reason that these linear methods are failing is due to the extreme seasonality and non-linearity present in the model. Although the Q value is dropped compared to the ARMA model, the MSE values are far greater than other models. Also Variance of 64.43 makes the estimators to be biased. 

![ARIMA Results](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/arma_model_res.JPG)

Considering MSE and variance of the prediction error, we could say Holts winter is working good for the dataset. Also, residual function from the one step prediction is almost equal to white noise pattern. Hence, performance of the holt’s winter is best on all the forecast models experimented for the given dataset. 

### h step Prediction - Holts Winter Method

![Holts_winter](https://github.com/gayuc07/Weather-Forecasting/blob/main/Images/holts_winer_h_setp.JPG)

## Summary & Conclusions

Thus, we could say that Holt winter method fits our problem with least MSE values and variance on the prediction error, making it best among other models. The same procedure must be followed for other seasonal data splits. The major reason for drop in the performance by linear methods like ARMA, ARIMA, SARIMA is because of the non-linearity present in the model along with multiple seasonal patterns and cyclic behavior of the dataset. This makes it difficult for the linear methods. For future scope, we may want to explore other non-linear models like transfer function or neural nets to improve the performances. 











