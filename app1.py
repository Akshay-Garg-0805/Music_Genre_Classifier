import base64
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import torch
import torchaudio
import plotly.graph_objects as go
import gdown
import tempfile

# Set up page configuration
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎶", layout="centered")

# Model download and loading functions
def download_model():
    url = "https://drive.google.com/uc?id=1jbzhth2qDgOPH624yGOfG5IbGXMdNO2P"
    output = "Trained_model_final.h5"
    gdown.download(url, output, quiet=False)

def load_model():
    download_model()
    model = tf.keras.models.load_model("Trained_model_final.h5")
    return model

# Audio processing
def load_and_preprocess_file(file_path, target_shape=(150,150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration, overlap_duration = 4, 2
    chunk_samples, overlap_samples = chunk_duration * sample_rate, overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start, end = i * (chunk_samples - overlap_samples), i * (chunk_samples - overlap_samples) + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(torch.tensor(chunk).unsqueeze(0)).numpy()
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data).reshape(-1, target_shape[0], target_shape[1], 1)

# Predict function
def model_prediction(x_test):
    model = load_model()
    y_pred = model.predict(x_test)
    
    # Ensure there are predictions and counts
    if y_pred.size == 0:
        st.error("Error: The model did not return any predictions.")
        return [], [], None
    
    predicted_cats = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_cats, return_counts=True)
    
    # Handle empty counts array
    if counts.size == 0:
        st.error("Error: No valid predictions returned.")
        return [], [], None
    
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    
    return unique_elements, counts, max_elements[0]
# Display pie chart
def show_pie(values, labels, test_mp3):
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    genre_labels = [classes[i] for i in labels]
    
    fig = go.Figure(
        go.Pie(
            labels=genre_labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            insidetextorientation='radial',
            hoverinfo='label+percent+value',
            pull=[0.15 if i == np.argmax(values) else 0.02 for i in range(len(values))],
            textfont=dict(family="Arial, sans-serif", size=14, color="white", weight="bold")
        )
    )
    
    fig.update_layout(
        title_text=f"🎼 Music Genre Classification: {test_mp3.name}",
        title_x=0.5,
        height=600,
        width=600,
        legend=dict(font=dict(family="Arial, sans-serif", size=16, color="white")),
        plot_bgcolor="#1f2c34",
        paper_bgcolor="#1f2c34",
        font_color="white"
    )
    
    st.plotly_chart(fig)

# Audio playback
def play_audio(test_mp3):
    audio_bytes = test_mp3.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_html = f"""
    <audio autoplay controls>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Global styles and layout adjustments
st.markdown(
    """
    <style>
        .footer, .header {
            width: 100%;
            text-align: left;
            color: white;    
        }
        .footer { position: fixed; bottom: 0; background-color: #0E1117; }
        .header { color: #00BFFF; font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
        .sidebar .sidebar-content { background-color: #1f2c34; color: white; }
        .toast-icon {
            position: fixed; right: 15px; bottom: 15px; width: 50px; height: 50px;
            background-color: #32CD32; border-radius: 50%; color: white; font-size: 1.5rem;
            display: flex; align-items: center; justify-content: center; cursor: pointer;
        }
        </style>
    """,
    unsafe_allow_html=True
)

# Toast icon navigation
st.markdown(
    """
    <div class="toast-icon" onclick="window.location.href='#predict-page'">🎵</div>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation 🚀")
app_mode = st.sidebar.radio("Go to", ["Home 🏠", "How it Works? ⚙️", "Predict Music Genre 🎧"])

# Render the header for the selected page
if app_mode == "Home 🏠":
    st.markdown('<div class="header">🎶 About Music Genre Classifier 🎶</div>', unsafe_allow_html=True)
    st.image("music_genre_home.png", width=350)
    st.write(
        """
        Welcome to the **Music Genre Classifier**! This tool uses **Deep Learning** to predict music genres. 
        Upload a track to discover its genre and dive into insights with our interactive visuals.
        """
    )

elif app_mode == "How it Works? ⚙️":
    st.markdown('<div class="header">🔍 How Does It Work?</div>', unsafe_allow_html=True)
    st.write(
        """
        The **Music Genre Classifier** is designed to predict the genre of a music track using advanced deep learning techniques. 
        Here’s a breakdown of how it works:

        ### 🎵 1. Upload Your Music
        Start by uploading an audio file in MP3 format. The system extracts small chunks from the audio to capture detailed features. 
        This helps in capturing the unique patterns present across different parts of the song.

        ### 🎼 2. Audio Feature Extraction
        - **Mel Spectrogram Generation**: Each audio chunk is transformed into a mel spectrogram, a visual representation of sound that 
          captures the intensity and frequency distribution over time. This process highlights elements such as rhythm, melody, and pitch 
          that vary among genres.
        - **Preprocessing for Consistency**: The spectrogram is then resized to a standard shape to ensure compatibility with the neural 
          network, making it easier for the model to analyze and classify.

        ### 🤖 3. Deep Learning Classification
        The **deep learning model** processes the spectrograms to identify genre characteristics. It outputs probabilities for each genre, 
        allowing us to display the top genre predictions.

        ### 📊 4. Visualize Results
        The genre prediction is displayed as an interactive pie chart, where each slice represents a genre and its probability.
        """
    )

elif app_mode == "Predict Music Genre 🎧":
    st.markdown('<div class="header" id="predict-page">🎶 Predict Music Genre</div>', unsafe_allow_html=True)
    st.write("Upload an audio file (mp3 format) 🎵")

    test_mp3 = st.file_uploader('', type=['mp3'])

    if test_mp3 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(test_mp3.getbuffer())
            filepath = tmp_file.name
            st.success(f"File {test_mp3.name} uploaded successfully! 🎉")

        if st.button("Play Audio 🔊"):
            play_audio(test_mp3)

        if st.button("Know Genre 🎼"):
            play_audio(test_mp3)
            with st.spinner("Analyzing... 🎶"):
                X_test = load_and_preprocess_file(filepath)
                labels, values, c_index = model_prediction(X_test)
                st.balloons()
                show_pie(values, labels, test_mp3)
    else:
        st.error("No file uploaded 🚫")

# Footer
st.markdown(
    """
    <div class="footer">© 2024 Music Genre Classifier | Enjoy Your Tunes! 🎶</div>
    """,
    unsafe_allow_html=True
)
