import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

# Custom LSTM wrapper to handle legacy parameters
def lstm_with_kwargs(**kwargs):
    kwargs.pop('time_major', None)  # Remove problematic argument
    return LSTM(**kwargs)

# Configure the app
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction using LSTM")

# Load model with custom handling
@st.cache_resource  # Cache model to avoid reloading
def load_custom_model(path):
    return load_model(path, custom_objects={'LSTM': lstm_with_kwargs})

model = load_custom_model('lstm_stock_model.h5')

def preprocess_data(data, look_back=60):
    # Select and sort data
    df = data[['Date', 'Open', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)  # Ensure chronological order

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Open', 'Close']])
    
    # Create sequences
    X = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :])
    X = np.array(X)
    
    return X, scaler, df

def plot_predictions(actual, predicted):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(actual, label='Actual Close Price', color='blue')
    ax.plot(predicted, label='Predicted Close Price', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# File upload
uploaded_file = st.file_uploader("Upload stock data CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    try:
        # Preprocess data
        look_back = 60
        X, scaler, df = preprocess_data(data, look_back)

        if len(X) == 0:
            st.error("Not enough data to make predictions. Need at least 60 days of historical data.")
        else:
            # Make predictions
            predictions = model.predict(X)

            # Handle different model output scenarios
            if predictions.shape[-1] == 2:  # Model predicts both Open and Close
                # Directly use model outputs for both features
                prediction_features = predictions
            else:  # Model predicts only Close
                # Create feature array with dummy Open values
                dummy_opens = X[:, -1, 0].reshape(-1, 1)
                prediction_features = np.hstack((dummy_opens, predictions))

            # Inverse transform using all features
            predictions = scaler.inverse_transform(prediction_features)[:, -1]  # Get Close prices

            # Get actual prices
            actual = df['Close'].values[look_back:]

            # Create date index for plotting
            dates = df.index[look_back:]

            # Display results
            st.subheader("Stock Price Predictions")
            plot_predictions(pd.Series(actual, index=dates), 
                            pd.Series(predictions, index=dates))

            # Show latest prediction
            latest_pred = predictions[-1]
            actual_price = actual[-1]
            st.metric(label="Next Day Prediction", 
                    value=f"${latest_pred:.2f}",
                    delta=f"{(latest_pred - actual_price):.2f} vs Current")

    except KeyError as e:
        st.error(f"Missing required column in CSV: {e}")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.sidebar.markdown("""
**Note:** The CSV file must contain these columns:
- Date (format: YYYY-MM-DD)
- Adj Close
- Close 
- High
- Low
- Open
- Volume
""")
st.write("Developed by Kumari Sweety Pandit")
