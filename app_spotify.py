import streamlit as st
import requests
import tempfile
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Helper functions
def track_names_to_ids(track_names, track_to_id):
    """
    Converts a sequence of track names to track IDs using the `track_to_id` mapping.
    """
    track_ids = [track_to_id.get(name, -1) for name in track_names]
    if -1 in track_ids:
        raise ValueError("One or more track names are not present in the `track_to_id` mapping.")
    return track_ids

def prepare_input_sequence(track_ids, max_sequence_length):
    """
    Pads the sequence of track IDs to match the `max_sequence_length` expected by the model.
    """
    return pad_sequences([track_ids], maxlen=max_sequence_length, padding='pre')

def get_track_uri_from_id(track_id, id_to_track):
    """
    Maps the predicted track ID to its track URI using the `id_to_track` mapping.
    """
    return id_to_track.get(track_id, "Unknown Track URI")

def predict_next_track(model, track_names, track_to_id, id_to_track, max_sequence_length):
    """
    Predicts the next track for a given sequence of track names.
    """
    # Convert track names to IDs
    track_ids = track_names_to_ids(track_names, track_to_id)
    
    # Prepare the input sequence
    input_sequence = prepare_input_sequence(track_ids, max_sequence_length)
    
    # Predict the next track
    predicted_probs = model.predict(input_sequence)
    predicted_track_id = predicted_probs.argmax(axis=1)[0]
    
    # Map the predicted track ID back to the track URI
    predicted_track_uri = get_track_uri_from_id(predicted_track_id, id_to_track)
    
    return predicted_track_uri

# Streamlit app
def main():
    st.title("Track Recommendation System")
    st.write("Enter a sequence of tracks to predict the next best track.")

    # Input: Sequence of tracks
    input_tracks = st.text_area(
        "Enter track names (comma-separated):",
        value="Hypnotize - 2014 Remastered Version, Big Poppa"
    )
    
    # Load mappings
    track_to_id = {
        "Hypnotize - 2014 Remastered Version": 0,
        "Big Poppa": 1,
        # Add more tracks here...
    }
    id_to_track = {v: k for k, v in track_to_id.items()}
    max_sequence_length = 10

    # Download and load the model
    model_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/rnn_model_Final.h5"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        model_path = temp_file.name
        response = requests.get(model_url)
        if response.status_code == 200:
            temp_file.write(response.content)
            st.success("Model downloaded successfully.")
        else:
            st.error("Failed to download model. Please check the URL.")
            return

    model = load_model(model_path)

    # Predict the next track
    if st.button("Predict Next Track"):
        try:
            # Convert input tracks to list
            track_names = [track.strip() for track in input_tracks.split(",")]

            # Predict next track
            predicted_track_uri = predict_next_track(
                model=model,
                track_names=track_names,
                track_to_id=track_to_id,
                id_to_track=id_to_track,
                max_sequence_length=max_sequence_length
            )
            st.success(f"Predicted next track: {predicted_track_uri}")
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
