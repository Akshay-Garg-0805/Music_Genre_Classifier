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

# Function to download and load model
def download_model():
    url = "https://drive.google.com/uc?id=1jbzhth2qDgOPH624yGOfG5IbGXMdNO2P"
    output = "Trained_model_final.h5"
    gdown.download(url, output, quiet=False)

def load_model():
    download_model()
    model = tf.keras.models.load_model("Trained_model_final.h5")
    return model

# Preprocessing
def load_and_preprocess_file(file_path, target_shape=(150,150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration, overlap_duration = 4, 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(torch.tensor(chunk).unsqueeze(0)).numpy()
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data).reshape(-1, target_shape[0], target_shape[1], 1)

# Prediction
def model_prediction(x_test):
    model = load_model()
    y_pred = model.predict(x_test)
    predicted_cats = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_cats, return_counts=True)
    return unique_elements, counts, unique_elements[counts.argmax()]

# Visualization
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
        title_text=f"Music Genre Classification: {test_mp3.name}",
        title_x=0,
        height=600,
        width=600,
        legend=dict(font=dict(family="Arial, sans-serif", size=16, color="white")),
        font_color="white"
    )
    
    st.plotly_chart(fig)

# Audio Playback
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

# Custom Styles and Toolbar
st.markdown(
    """
    <style>
        .toolbar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            background-color: #0E1117;
            padding: 1rem 0;
        }
        .toolbar a {
            color: #00BFFF;
            font-size: 1.2rem;
            text-decoration: none;
            font-weight: bold;
        }
        .toolbar a:hover {
            color: #32CD32;
            text-decoration: underline;
        }
        .footer, .toast-icon {
            position: fixed;
            width: 100%;
            background-color: #0E1117;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
        .footer {
            bottom: 0;
        }
        .toast-icon {
            right: 15px;
            bottom: 15px;
            width: 50px;
            height: 50px;
            background-color: #32CD32;
            border-radius: 50%;
            font-size: 1.5rem;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Toolbar and clickable toast for navigation
st.markdown(
    """
    <div class="toolbar">
        <a href="#home">Home</a>
        <a href="#about-app">About App</a>
        <a href="#how-it-works">How It Works</a>
        <a href="#predict-music-genre">Predict Music Genre</a>
    </div>
    <div class="toast-icon" onclick="window.location.href='#predict-music-genre';">🎵</div>
    """,
    unsafe_allow_html=True
)

# Page Rendering
page_section = st.experimental_get_query_params().get("page", ["Home"])[0]

if page_section == "Home":
    st.title("🎶 Welcome to the Music Genre Classifier 🎶")
    st.image("music_genre_home.png", width=350)
    st.write("""
        **This app uses deep learning to classify your favorite music tracks into genres!**  
        Upload a track and let AI reveal its genre, providing insights with intuitive visuals.
    """)

elif page_section == "About App":
    st.title("About the App")
    st.write("""
        The Music Genre Classifier is an AI-powered tool that automatically analyzes music tracks and classifies them into genres.
    """)

elif page_section == "How it Works":
    st.title("How It Works")
    st.write("""
        1. **Upload a Music File**: Start by uploading an audio file in mp3 format.
        2. **Genre Prediction**: The AI processes the audio and predicts the genre distribution.
        3. **Explore Genres**: Discover the genre distribution in your music with a pie chart.
    """)

elif page_section == "Predict Music Genre":
    st.title("Predict Music Genre")
    st.markdown("##### Upload an audio file (mp3 format)")
    test_mp3 = st.file_uploader('', type=['mp3'])

    if test_mp3 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(test_mp3.getbuffer())
            filepath = tmp_file.name
            st.success(f"File {test_mp3.name} uploaded successfully!")

        if st.button("Play Audio"):
            play_audio(test_mp3)

        if st.button("Know Genre"):
            play_audio(test_mp3)
            with st.spinner("Please wait..."):
                X_test = load_and_preprocess_file(filepath)
                labels, values, c_index = model_prediction(X_test)
                st.balloons()
                show_pie(values, labels, test_mp3)
    else:
        st.error("No file uploaded")

# Footer
st.markdown(
    """
    <div class="footer">© 2024 Music Genre Classifier | Enjoy Your Tunes! 🎶</div>
    """,
    unsafe_allow_html=True
)
