import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import torch
import torchaudio
from matplotlib import pyplot as plt

# Load model
st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("./Trained_model_final.h5")
    return model

# Preprocess source file
def load_and_preprocess_file(file_path, target_shape=(150, 150)):
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
        
        # Convert chunk to Mel Spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(torch.tensor(chunk).unsqueeze(0)).numpy()
        
        # Resize matrix based on provided target shape (150, 150)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

        # Remove extra dimensions, the shape should be (150, 150, 1)
        mel_spectrogram = np.squeeze(mel_spectrogram, axis=0)  # Remove the first dimension
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Ensure the shape is (150, 150, 1)
        
        data.append(mel_spectrogram)
    
    return np.array(data)

# Predict values
def model_prediction(x_test):
    model = load_model()
    y_pred = model.predict(x_test)
    predicted_cats = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_cats, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return unique_elements, counts, max_elements[0]

# Show pie chart
def show_pie(values, labels, test_mp3):
    plt.figure(figsize=(8,8))
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    label = [classes[i] for i in labels]
    max_index = np.argmax(values)
    explode = [0.2 if i == max_index else 0 for i in range(len(values))]

    plt.pie(values, labels=label, autopct='%1.1f%%', startangle=140, explode=explode)
    plt.title("Music: "+test_mp3.name)
    
    st.pyplot(plt)

# Sidebar UI
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["About app", "How it works?", "Predict music genre"])

# Main page
if app_mode == "About app":
    st.markdown(
        """
        <style>
        .stapp {
            background-color: #0E1117;
            color: white;
        }
        h2 {
            color: #00BFFF; /* Deep Sky Blue for main title */
            font-size: 36px;
            font-weight: bold;
        }
        h3 {
            color: #ADD8E6; /* Light Blue for subtitles */
            font-size: 28px;
        }
        p {
            color: #D3D3D3; /* Light Gray for text */
            font-size: 18px;
        }
        .stmarkdown {
            color: #D3D3D3;
        }
        .stimage {
            border-radius: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('''## Welcome to the,''')
    st.markdown('''## Music Genre Classifier 🎶🎧''')
    image_path = "music_genre_home.png"
    st.image(image_path, width=350)

    st.markdown("""
    ## Welcome to the Music Genre Classifier, an AI-powered app designed to help you explore and categorize music with ease! ✨
                
    Leveraging deep learning (DL) techniques, this app automatically analyzes your music tracks and classifies them into various genres with impressive accuracy.

    Whether you’re a music enthusiast, a DJ, or just someone who loves discovering new tunes, this app brings AI to your fingertips, transforming how you interact with music. Simply upload a song, and within seconds, our advanced model will determine the genre, providing you with a seamless and intuitive music discovery experience.

    ### **Key Features: 💪**

    AI-Powered Music Classification: Built using deep learning, our app accurately classifies music into multiple genres.
                
    Fast & Easy: Upload a track, and let the app work its magic in seconds.
                
    Explore New Music: Find genres you might not have explored before and discover new favorites.
                
    User-Friendly Interface: An easy-to-use design makes classifying your music a breeze.
                
    #### Let our deep learning model do the heavy lifting while you enjoy the world of music in a whole new way! 🎧

    (_P.S. -> It is an AI model so it may give wrong predictions too_)
    """)

elif app_mode == "How it works?":
    st.markdown("""
    # How to know the music genre?
    **1. Upload music: Start off with uploading the music file**
    **2. Analysis: Our system will process the music file with advanced algorithms to classify it into a number of genres**
    **3. Results: After the analysis phase, you will get a pie chart depicting the percentage of genres the music belongs to (A music is not purely a single genre)**

    #### _P.S. -> You can also listen to the music in the app itself_
    """)

elif app_mode == 'Predict music genre':
    st.header("**_Predict Music Genre_**")
    st.markdown('##### Upload the audio file (mp3 format)')
    test_mp3 = st.file_uploader('', type=['mp3'])

    if test_mp3 is not None:
        filepath = "test_data/" + test_mp3.name
        with open(filepath, 'wb') as f:
            f.write(test_mp3.getbuffer())
        st.success(f"File {test_mp3.name} uploaded successfully!")

    # Play audio
    if st.button("Play Audio") and test_mp3 is not None:
        st.audio(test_mp3)

    # Predict
    if st.button("Know Genre") and test_mp3 is not None:
        with st.spinner("Please wait ..."):
            X_test = load_and_preprocess_file(filepath)
            labels, values, c_index = model_prediction(X_test)

            st.balloons()
            st.markdown("The music genre is : ")
            show_pie(values, labels, test_mp3)
