import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Set up Streamlit
st.title("Stock Price Predictor App")
st.subheader("By Kosemani abdulalim Adeshina 20/47cs/01133")

# Get user input for stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define time range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end)

# Load the model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Define function for plotting
def plot_graph(values, full_data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_data.index, y=full_data.Close, mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=full_data.index, y=values, mode='lines', name=title))
    return fig

# Display Moving Averages
st.subheader('Original Close Price and Moving Averages')
google_data['MA_250'] = google_data.Close.rolling(250).mean()
google_data['MA_200'] = google_data.Close.rolling(200).mean()
google_data['MA_100'] = google_data.Close.rolling(100).mean()

fig_250 = plot_graph(google_data['MA_250'], google_data, 'MA_250')
fig_200 = plot_graph(google_data['MA_200'], google_data, 'MA_200')
fig_100 = plot_graph(google_data['MA_100'], google_data, 'MA_100')

st.plotly_chart(fig_250, use_container_width=True)
st.plotly_chart(fig_200, use_container_width=True)
st.plotly_chart(fig_100, use_container_width=True)

# Prepare data for prediction
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict stock prices
predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data.reshape(-1, 1))

# Prepare data for plotting
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# Display prediction results
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot original vs predicted values
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=google_data.index[:splitting_len + 100], y=google_data.Close[:splitting_len + 100], mode='lines', name='Training Data'))
fig_pred.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['original_test_data'], mode='lines', name='Original Test Data'))
fig_pred.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['predictions'], mode='lines', name='Predicted Test Data'))
fig_pred.update_layout(title='Original Close Price vs Predicted Close Price')

st.plotly_chart(fig_pred, use_container_width=True)

# Future prediction
future_days = st.slider('Number of days to predict into the future', 1, 30, 7)

# Predict future prices
last_100_days = google_data.Close[-100:].values
last_100_days_scaled = scaler.transform(last_100_days.reshape(-1, 1))

X_test_future = [last_100_days_scaled]

predicted_future_prices = []
for _ in range(future_days):
    pred_price = model.predict(np.array(X_test_future).reshape(1, -1, 1))
    predicted_future_prices.append(pred_price)
    new_input = np.append(X_test_future[0][1:], pred_price).reshape(-1, 1)
    X_test_future = [new_input]

predicted_future_prices = np.array(predicted_future_prices).reshape(-1, 1)
predicted_future_prices = scaler.inverse_transform(predicted_future_prices)

# Create future dates
future_dates = pd.date_range(start=google_data.index[-1] + pd.DateOffset(1), periods=future_days)

# Plot future predictions
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=google_data.index, y=google_data.Close, mode='lines', name='Historical Data'))
fig_future.add_trace(go.Scatter(x=future_dates, y=predicted_future_prices.reshape(-1), mode='lines', name='Future Predictions'))
fig_future.update_layout(
    title=f'Future Stock Price Prediction for {future_days} days',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis=dict(
        tickformat='%Y-%m-%d'  # Format for the dates
    )
)

st.plotly_chart(fig_future, use_container_width=True)
