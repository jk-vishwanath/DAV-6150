import streamlit as st
from keras.models import load_model
import pandas as pd
import numpy as np

def load_data():
    # Load your dataset
    try:
        data = pd.read_csv("https://github.com/jk-vishwanath/DAV-6150/raw/refs/heads/main/df_merge.csv")  # Update with the correct path
        return data
    except FileNotFoundError:
        st.error("Dataset file not found!")
        return None

def preprocess_input(track_uri, data):
    # Validate and extract track features
    if track_uri not in data['track_uri'].values:
        st.error("Unknown track URI. Please provide a valid Spotify track URI.")
        return None

    # Extract features for the given track URI
    features = data.loc[data['track_uri'] == track_uri, ['feature1', 'feature2', ...]].values
    return features

def main():
    st.title("Spotify Track Predictor")

    # Input: Track URI
    track_uri = st.text_input("Enter a Spotify track URI:")
    if not track_uri:
        st.warning("Please enter a track URI to continue.")
        return

    # Load dataset
    dataset = load_data()
    if dataset is None:
        return  # Exit if dataset loading fails

    # Preprocess input
    features = preprocess_input(track_uri, dataset)
    if features is None:
        return  # Exit if preprocessing fails

    # Load the model
    model_path = "model.h5"  # Update with the correct model path
    try:
        model = load_model('https://github.com/jk-vishwanath/DAV-6150/raw/refs/heads/main/rnn_model_Final.h5')
    except OSError:
        st.error("Failed to load the model. Please check the file path and format.")
        return

    # Predict the next track
    if st.button("Predict Next Track"):
        prediction = model.predict(features)
        predicted_track_index = np.argmax(prediction)  # Example if using classification
        predicted_track = dataset.iloc[predicted_track_index]['track_name']

        st.success(f"Predicted next track: {predicted_track}")

if __name__ == "__main__":
    main()
