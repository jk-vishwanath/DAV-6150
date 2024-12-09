import streamlit as st
from keras.models import load_model
import pandas as pd
import numpy as np
import os
import requests

# Cache dataset loading for efficiency
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(
            "https://github.com/jk-vishwanath/DAV-6150/raw/refs/heads/main/df_merge.csv"
        )
        return data
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

# Cache model loading
@st.cache_resource
def load_model_from_url(model_url, save_path):
    if not os.path.exists(save_path):
        try:
            response = requests.get(model_url, stream=True)
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download the model: {e}")
            return None

    try:
        model = load_model(save_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Preprocess input
def preprocess_input(track_track_name, data):
    if track_track_name not in data['track_track_name'].values:
        st.error("Unknown track name. Please provide a valid Spotify track name.")
        return None

    # Extract features (track position in this case)
    features = data.loc[data['track_track_name'] == track_track_name, ['track_pos']].values
    # Reshape to match model input shape: (batch_size, sequence_length, feature_dim)
    features = features.reshape(1, 1, -1)
    return features

# Main Streamlit app
def main():
    st.title("Spotify Track Predictor")

    # Input: Track Name
    track_name = st.text_input("Enter a Spotify track name:")
    if not track_name:
        st.warning("Please enter a track name to continue.")
        return

    # Load dataset
    dataset = load_data()
    if dataset is None:
        return  # Exit if dataset loading fails

    # Preprocess input
    features = preprocess_input(track_name, dataset)
    if features is None:
        return  # Exit if preprocessing fails

    # Load the model
    model_url = 'https://github.com/jk-vishwanath/DAV-6150/raw/refs/heads/main/rnn_model_Final.h5'
    save_path = 'rnn_model_Final.h5'
    model = load_model_from_url(model_url, save_path)
    if model is None:
        return  # Exit if model loading fails

    # Predict the next track
    if st.button("Predict Next Track"):
        try:
            prediction = model.predict(features)
            predicted_track_index = np.argmax(prediction)  # Assuming classification model
            predicted_track = dataset.iloc[predicted_track_index]['track_track_name']
            st.success(f"Predicted next track: {predicted_track}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
