import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Lotto 4D Prediction Bot")

# File uploader
data_file = st.file_uploader("Upload your Lotto 4D CSV file", type=["csv"])

if data_file:
    # Load dataset
    data = pd.read_csv(data_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Rename columns if needed
    data.columns = ['Index', 'Date', 'Draw1', 'Draw2', 'Draw3', 'Draw4']

    # Preprocess data
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
    data = data.drop(columns=['Index'])
    st.subheader("Cleaned Data")
    st.write(data.head())

    # Feature and target preparation
    X = data[['Draw1', 'Draw2', 'Draw3', 'Draw4']].values
    y = data['Draw1'].shift(-1).fillna(0).astype(int).values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions and accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    # Frequency analysis
    all_numbers = data[['Draw1', 'Draw2', 'Draw3', 'Draw4']].values.flatten()
    number_counts = pd.Series(all_numbers).value_counts()
    top_numbers = number_counts.head(4).index.tolist()

    # Suggestions
    st.subheader("Number Suggestions")
    st.write("Based on Frequency Analysis:", top_numbers)
    st.write("Based on Model Prediction:", y_pred[:4].tolist())
