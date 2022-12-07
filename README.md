# Stock-Price-Prediction-with-Keras
[Medium Post](https://medium.com/@zahmed333/stock-price-prediction-with-keras-df87b05e5906)
## Authors

[**_Kevin Shah_**] [**_Zulnorain Ahmed_**] [**_Hayden Snyder_**]

![](https://miro.medium.com/max/1400/1*LXWfUlbziH1B3jh1UCb3Lg.jpeg)

## Background

## Motivation

The financial technology industry is rapidly growing and we wanted to apply the skills we’ve learned during our machine learning course. This would allow us to gain some experience in an area of interest and make us more competitive candidates.

## What is a LSTM model?

Long short-term memory (LSTM) is a deep learning architecture based on a recurrent neural network (RNN). This architecture is great to use for time-series predictions which in our case is favorable since that is involved with stock modeling. The architecture was specifically designed to solve the vanishing gradient problem faced by RNNs. It does this through feedback connections that enable it to process entire sequences of data.

In general, LSTM is used for predicting and generating predictions. For instance, it would be good at understanding hand-writing and speech recognition. There are three gates in a LSTM cell that make up its architecture.

![](https://miro.medium.com/max/1400/1*Z4k74pkkpEgfpEvbbUZuAg.png)

LSTM Cell

The first is the **forget gate** which gets to decide which piece of information goes out and which piece needs attention. The information from the addition of new information X(t) and hidden state H(t-1) is passed through the sigmoid function. Then, the sigmoid generates values between 0 and 1. Which determines what will happen to the information. If it is necessary, the output will be closer to 1.

Secondly, the **input gate** updates the cell status using the following operations. First, the current state X(t) and previously hidden state H(t-1) are passed into the second sigmoid function. The values are transformed between important to not important (0–1). Next, the same information of the hidden state and current state will be passed through the tanh function. To regulate the network, the tanh operator will create a vector (C~(t) ) with all the possible values between -1 and 1. The output values generated form the activation functions.

Finally, the **output gate** determines useful information from the cell state and outputs that. The values of the current state and previous hidden state are passed into the third sigmoid function. Then the new cell state generated from the cell state is passed through the tanh function. Both these outputs are multiplied point-by-point. Based upon the final value, the network decides which information the hidden state should carry. This hidden state is used for prediction.

## The Data

The two data sources we debated were kaggle and yfinance, we ended up going with the latter because kaggle didn’t have data that was sorted for us to use easily. In order to train and test the model, we used an open source library called yfinance. This library has collected various aspects of stocks since 1962, including the stock prices, news headlines, financial reports and company information. It is simple to retrieve data with the yfinance API. Using each stock’s ticker, we could store the data in a variable. Then, we gathered the stock price history for 10 stocks across various industries.

To do this we first had to instantiate the ticker for those 10 stocks to pull the stock price.  
\*NOTE: This is a run through on what the Amazon stock would look like:

```
import yfinance as yffrom datetime import datetimeamzn = yf.Ticker("AMZN")end_date = datetime.now().strftime('%Y-%m-%d')amzn_hist = amzn.history(start='2017-01-01',end=end_date)print(amzn_hist)
```

Which in turn outputs:

![](https://miro.medium.com/max/1400/1*yl8AW05R67YBm45PK4YRiw.png)

Amazon Stock history from 2017–2022

As you can see from the example above, We have plenty of data available for the stock and ranging from the opening price to the volume on that specific day. This would be a lot of features to account for, therefore, we are only taking the closing price of the stock on the given day of the market. We then make sure the only data we want is the closing price.

```
amzn_close = amzn_hist['Close']amzn_values = amzn_close.valuesamzn_values = amzn_values.reshape(-1,1)
```

After indexing to obtain only the close prices, we then have to scale the data between 0 and 1. This is called a MinMax scaler which should be used when it is required to capture small variance in features. This transformation of data being fitted allows us to see the minute differences between the stock prices on days where the stock doesn’t move much.

```
trainingScaler = MinMaxScaler(feature_range=(0,1))amzn_values_scaled = trainingScaler.fit_transform(amzn_values)print(len(amzn_values_scaled))
```

## The Model

## Training

Once we gathered the stock history we needed and cleaned it, we were able to create training data. In the code below, we created our training and testing data. We had already mapped the date to the price in previous steps thanks to the yfinance API.

The criteria we went with was the past 5 years for the closing prices. We divided five years of each stocks closing prices into training and testing data We divided it up with 85% for training, 15% for testing. The training data was divided into batches of 50, where we predicted on the 51st day. The following code shows how we performed it on Amazon.

```
training_split = math.floor(len(amzn_values_scaled) * 0.85) training_amzn = amzn_values_scaled[0:training_split]training_ind_amzn = []for i in range(50, len(training_amzn)):  training_ind_amzn.append(training_amzn[i-50:i][0])  training_dep_amzn.append(training_amzn[i][0])training_ind_amzn, training_dep_amzn = np.array(training_ind_amzn), np.array(training_dep_amzn)training_ind_amzn = np.reshape(training_ind_amzn, (training_ind_amzn.shape[0], training_ind_amzn.shape[1], 1))
```

We are making an independent batch for training data as well as dependent training data.

## Architecture

Now that the training and test data is organized and divided, we can construct the network’s architecture. This is a sequential model that allows us to create an architecture layer-by-layer. Each layer has exactly one input tensor and one output tensor. We create a model for each stock and add several layers of 100 units with return\_sequences set to “True” so that the output sequence maintains the same length. Specifically, we are using 3 LSTM layers with 100 nodes each. We then use dropout layers to drop 20 percent of the layers. Then, we add two densely connected layers, the second of which specifies a single output. We run this model for a total of 60 epochs and a batch size of 32 with the Adam optimizer.

```
amzn_model = Sequential()amzn_model.add(LSTM(100, return_sequences=True, input_shape=(training_ind_amzn.shape[1], 1)))amzn_model.add(Dropout(0.2))amzn_model.add(LSTM(100, return_sequences=True))amzn_model.add(Dropout(0.2))amzn_model.add(LSTM(100))amzn_model.add(Dropout(0.2))amzn_model.add(Dense(25))amzn_model.add(Dense(1))amzn_model.compile(optimizer='adam',loss='mean_squared_error')amzn_model.fit(training_ind_amzn, training_dep_amzn, epochs = 60, batch_size = 32)
```

## Testing

We are testing the dataset here by transforming the testing input. Then for the 50 days we are using as training, we are then adding that data to the stock as input. Once it goes through 50 iterations, we shape the data and then predict using our model.

```
testing_input_amzn = amzn_values[training_split:]testing_input_amzn = trainingScaler.fit_transform(testing_input_amzn)testing_amzn = []for i in range(50, len(testing_input_amzn) + 50):  testing_amzn.append(testing_input_amzn[i-50:i][0])testing_amzn = np.array(testing_amzn)testing_amzn = np.reshape(testing_amzn, (testing_amzn.shape[0], testing_amzn.shape[1], 1))predict_amzn = amzn_model.predict(testing_amzn)predict_amzn = trainingScaler.inverse_transform(predict_amzn)
```

## Results

## Plotting Stocks

We utilized matplotlib.pyplot to plot our data that we had trained to show the difference between predicted price and actual price.

Blue = Actual Stock Price  
Red = Predicted Stock Price

```
plt.plot(amzn_values[training_split:], color = 'blue', label = 'AMZN Stock Price')plt.plot(predict_amzn, color = 'red', label = 'Predicted AMZN Stock Price')plt.title('Amazon (AMZN)')plt.xlabel('Number of Days since April 26, 2022')plt.ylabel('AMZN Stock Price')plt.legend()plt.show()
```

## Stock Graphs:

![](https://miro.medium.com/max/778/1*QBPFyF9qDTZS7EFtDWJWMQ.png)

Mean Average Percentage Error: 1.37%

![](https://miro.medium.com/max/778/1*HHB_YMaf97hH3DT5D9imlg.png)

Mean Average Percentage Error: 1.15%

![](https://miro.medium.com/max/778/1*Ey49ndq4iUZepFiQOFmCHg.png)

Mean Average Percentage Error: 7.63%

![](https://miro.medium.com/max/778/1*IYPUPGHIqWpre7s-4vO56Q.png)

Mean Average Percentage Error: 5.45%

![](https://miro.medium.com/max/778/1*WGeuvI1CpRugCWit7ewL5w.png)

Mean Average Percentage Error: 9.30%

![](https://miro.medium.com/max/778/1*IFZqtGxUaxLRLp9ickI5Ng.png)

Mean Average Percentage Error: 1.04%

![](https://miro.medium.com/max/778/1*VcEj8OpWOPQUNu_KtMpKRA.png)

Mean Average Percentage Error: 3.12%

![](https://miro.medium.com/max/802/1*Y9B5q3sF7aM6DhrCYKVpPg.png)

Mean Average Percentage Error: 6.54%

![](https://miro.medium.com/max/764/1*N5SqfnfktJ6I9SjuQi6reA.png)

Mean Average Percentage Error: 3.35%

![](https://miro.medium.com/max/784/1*lm_754AxqTPJ1wcgq9WovA.png)

Mean Average Percentage Error: 3.12%

![](https://miro.medium.com/max/1400/1*qjFxs3Upx2MyLtG84fgZpw.png)

We calculated the Mean Absolute Percentage Error (MAPE), the Root of Mean Squared Error (RMSE), and the percentage of days that the prediction was in the right direction on our models.

## Analysis

I want to start with defining some of the important terms that we introduced in above figure. The MAPE calculates the average accuracy as a percentage. It is good for forecast systems and larger numbers with fewer extremes. The RMSE indicates the absolute fit of the model. In other words, it shows how close the points are to the predicted values.

Taiwan Semiconductor (TSM) seems to be an outlier for all the values analyzed. TSM and TSLA have some of the highest values which makes sense because of how volatile their stocks are. Furthermore, Bitcoin is an extreme outlier which makes sense because of its extreme volatility with such highs and lows within its pricing for the past few years.

From these results we can conclude that volatility is what shakes up the model the most out of anything else.

## Most Accurate MAPE: McDonalds

Mean Average Percentage Error: 1.04%

The reasons for such a low percentage error could be that McDonalds is a blue-chip stock meaning it is a running company for a long time. The fact that the company has lasted multiple recessions and has stood the test of time. It was started in 1955, which could explain why.

## Least Accurate MAPE: Tesla

Mean Average Percentage Error: 9.30%

The reasons for such a high percentage error could be that Tesla is known for its volatility, and that is something that model can’t accurately account for. It was started in July 1, 2003 which is relatively recent, and is very popular on social media because it is a leader in the electric auto-vehicle market.

**Limitations of Using Neural Nets in Stock Price Predictions**

Price data (which we used to train our models) is a lagging indicator. This means that price tends to change after a trend has already started. Therefore, this phenomenon makes the predictions less accurate.

It may not provide enough information to accurately predict future conditions because some stocks have a 9% average accuracy

Furthermore, LSTM models do not accurately predict volatile, fluctuating stocks/derivatives. (e.g. tech stocks and crypto ) However, they do well with large-cap stocks in more stable industries (Retail/Ecommerce: Amazon, Walmart Food: McDonalds)

## **Future Additions**

The main addition we can try adding in are technical indicators (moving averages, Relative Strength Index, etc) to the training of the model. These are massively used within stock trading. If we were to add this into the model as different features I believe it would make the predictions more accurate.

Finding more granular data (minute/hourly) rather than the daily price data that we had to use. This could provide more usable predictions for people who want to use it for things like options trading.

Other than creating different models based on varying timeframes. Using a different neural network architecture. The following article describes another architecture that can provide more accurate results.

[https://www.sciencedirect.com/science/article/pii/S1877050919302789](https://www.sciencedirect.com/science/article/pii/S1877050919302789)

To be specific, a GAN with LSTM as generator and MLP as discriminator is being used. This is different than using just a LSTM, which is what we do in our model.

## Conclusion

Our results suggest that our model was relatively successful in in achieving the goal of predicting changes in the stock market, especially in stable industries like retail.

8 out of our 10 models predicted the direction of price change over 99.5 % of the time, which means that it could be useful for profits.

The models struggled with volatile markets in both price direction and accuracy.

It would be interesting to test out these prediction with a paper trading account to see if there is any profit made.
