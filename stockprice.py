import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("Stock Price Trend Prediction Bot")

# File uploader
data_file = st.file_uploader("Upload your Stock Prices CSV file", type=["csv"])

if data_file:
    # Load dataset
    data = pd.read_csv(data_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Rename columns if needed
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Preprocess data
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data = data.drop(columns=['Date'])
    st.subheader("Cleaned Data")
    st.write(data.head())

    # Feature and target preparation
    X = data[['Open', 'High', 'Low', 'Volume']].values
    y = data['Close'].shift(-1).fillna(data['Close'].iloc[-1]).values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predictions and performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Feature importance analysis
    feature_importance = pd.Series(model.feature_importances_, index=['Open', 'High', 'Low', 'Volume']).sort_values(ascending=False)

    # Display feature importance
    st.subheader("Feature Importance")
    st.bar_chart(feature_importance)

    # Future trend suggestions
    future_predictions = model.predict(data[['Open', 'High', 'Low', 'Volume']].tail(4).values)
    st.subheader("Future Trend Suggestions")
    st.write("Predicted Closing Prices:", future_predictions.tolist())
