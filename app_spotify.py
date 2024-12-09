import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
import requests

def load_data():
    """Load the dataset from the provided URL."""
    try:
        data = pd.read_csv(
            "https://github.com/jk-vishwanath/DAV-6150/raw/refs/heads/main/df_merge.csv"
        )
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def download_model(model_url, save_path):
    """Download the model file from a URL."""
    try:
        r = requests.get(model_url, stream=True)
        with open(save_path, "wb") as file:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except Exception as e:
        st.error(f"Error downloading model: {e}")

def load_model_from_url(model_url, save_path):
    """Load the model, downloading it if necessary."""
    if not os.path.exists(save_path):
        download_model(model_url, save_path)

    try:
        model = load_model(save_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(track_track_name, data):
    """Preprocess input to create a padded sequence for the model."""
    if track_track_name not in data['track_track_name'].values:
        st.error("Unknown track name. Please provide a valid track name.")
        return None

    # Convert track name to corresponding track position (or ID) as input
    track_pos = data.loc[data['track_track_name'] == track_track_name, 'track_pos'].values
    input_sequence = list(track_pos)

    # Pad the sequence to the required input length
    sequence_length = 2002  # Match the model's expected sequence length
    input_sequence_padded = pad_sequences([input_sequence], maxlen=sequence_length, padding='pre')
    return input_sequence_padded

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

    # Load model
    model_url = 'https://github.com/jk-vishwanath/DAV-6150/raw/refs/heads/main/rnn_model_Final.h5'
    save_path = 'rnn_model_Final.h5'
    model = load_model_from_url(model_url, save_path)
    if model is None:
        return  # Exit if model loading fails

    # Predict the next track
    if st.button("Predict Next Track"):
        try:
            # Perform prediction
            prediction = model.predict(features)
            predicted_track_index = np.argmax(prediction)
            predicted_track = dataset.iloc[predicted_track_index]['track_track_name']

            st.success(f"Predicted next track: {predicted_track}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
