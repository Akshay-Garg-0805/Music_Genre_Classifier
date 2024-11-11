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

# Set up the page configuration
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎶", layout="centered")

def download_model():
    url = "https://drive.google.com/uc?id=1jbzhth2qDgOPH624yGOfG5IbGXMdNO2P"
    output = "Trained_model_final.h5"
    gdown.download(url, output, quiet=False)

def load_model():
    download_model()
    model = tf.keras.models.load_model("Trained_model_final.h5")
    return model

def load_and_preprocess_file(file_path, target_shape=(150,150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
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

def model_prediction(x_test):
    model = load_model()
    y_pred = model.predict(x_test)
    predicted_cats = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_cats, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return unique_elements, counts, max_elements[0]

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
            textfont=dict(
                family="Arial, sans-serif",
                size=14,
                color="white",
                weight="bold"
            )
        )
    )
    
    fig.update_traces(
        hoverinfo="label+percent+value",
        hovertemplate="%{label}: %{value} songs (%{percent})",
        marker=dict(line=dict(color="#FFFFFF", width=2))
    )
    
    fig.update_layout(
        title_text=f"Music Genre Classification: {test_mp3.name}",
        title_x=0.5,
        height=600,
        width=600,
        legend=dict(
            font=dict(family="Arial, sans-serif", size=16, color="white"),
            title="Genres",
            title_font=dict(size=18, color="white")
        ),
        plot_bgcolor="#1f2c34",
        paper_bgcolor="#1f2c34",
        font_color="white"
    )
    
    st.plotly_chart(fig)

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

# Global styles and footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0E1117;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
        .header {
            color: #00BFFF;
            font-size: 2rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .toast-icon {
            position: fixed;
            right: 15px;
            bottom: 15px;
            width: 50px;
            height: 50px;
            background-color: #32CD32;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["About App", "How it Works?", "Predict Music Genre"])

if app_mode == "About App":
    st.markdown('<div class="header">About Music Genre Classifier</div>', unsafe_allow_html=True)
    st.markdown("Welcome to the Music Genre Classifier 🎶🎧")
    st.image("music_genre_home.png", width=350)
    st.markdown("""
        ## This AI-powered app categorizes music into genres with ease.
        - **AI-Powered Classification**: Uses DL techniques for accurate genre classification.
        - **Fast & Easy**: Upload a song and see results within seconds.
        - **User-Friendly Interface**: Makes classifying your music a breeze.
        #### Let AI classify your music and discover new genres!
    """)

elif app_mode == "How it Works?":
    st.markdown('<div class="header">How Does It Work?</div>', unsafe_allow_html=True)
    st.markdown("""
        1. **Upload music**: Start with uploading a music file.
        2. **AI Analysis**: The system processes the music and classifies it into genres.
        3. **See Results**: View a pie chart showing genre probabilities.

        _You can also listen to the music directly in the app!_
    """)

elif app_mode == 'Predict Music Genre':
    st.markdown('<div class="header">Predict Music Genre</div>', unsafe_allow_html=True)
    st.markdown('##### Upload an audio file (mp3 format)')
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

st.markdown(
    """
    <div class="footer">© 2024 Music Genre Classifier | Enjoy Your Tunes! 🎶</div>
    <div class="toast-icon">🎵</div>
    """,
    unsafe_allow_html=True
)
