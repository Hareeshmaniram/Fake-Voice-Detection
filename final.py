#importing libraries

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib 


def audio_to_melspectrogram(filepath, output_img_path):
    # Load audio file
    y, sr = librosa.load(filepath, duration=3, sr=22050)  # limit to 3 seconds
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Save the mel spectrogram as an image
    plt.figure(figsize=(2.24, 2.24))
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis('off')  # no axes for clarity
    plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

#  Prepare Data (Convert All Audio to Spectrograms) ---
def prepare_dataset(real_audio_dir, fake_audio_dir, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    labels = []
    data = []

    # Debugging: Check directories
    print(f"Processing real audio directory: {real_audio_dir}")
    print(f"Processing fake audio directory: {fake_audio_dir}")

    # Process real audio files (label = 0)
    for filename in os.listdir(real_audio_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(real_audio_dir, filename)
            img_path = os.path.join(output_img_dir, f"real_{filename.replace('.wav', '.png')}")
            audio_to_melspectrogram(file_path, img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) / 255.0  # Normalize image
            data.append(img.flatten())  # Flatten the image to a 1D array
            labels.append(0)  # Label '0' for real audio

    # Process fake audio files (label = 1)
    for filename in os.listdir(fake_audio_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(fake_audio_dir, filename)
            img_path = os.path.join(output_img_dir, f"fake_{filename.replace('.wav', '.png')}")
            audio_to_melspectrogram(file_path, img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) / 255.0  # Normalize image
            data.append(img.flatten())  # Flatten the image to a 1D array
            labels.append(1)  # Label '1' for fake audio

    # Check if dataset is empty
    if len(data) == 0 or len(labels) == 0:
        raise ValueError("The dataset is empty. Ensure the audio files are correctly processed.")

    return np.array(data), np.array(labels)

#  Train the Random Forest Classifier ---
def train_model(real_audio_dir, fake_audio_dir, output_img_dir):
    X, y = prepare_dataset(real_audio_dir, fake_audio_dir, output_img_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(clf, 'deepfake_audio_classifier_rf.pkl')
    print("Model trained and saved successfully!")
    return clf

#  Predict Real or Fake from a New Audio File ---
def predict_audio(filepath, model):
    # Convert audio to spectrogram
    img_path = 'temp_spectrogram.png'
    audio_to_melspectrogram(filepath, img_path)
    
    # Load and preprocess the spectrogram image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128)) / 255.0  # Normalize
    img = img.flatten()  # Flatten the image to a 1D array

    # Predict whether the audio is real or fake
    prediction = model.predict([img])
    if prediction == 1:
        print("The audio is FAKE.")
    else:
        print("The audio is REAL.")

#  Main Function to Execute Everything ---
if __name__ == "__main__":
    # Define paths based on the datasets
    validated_csv_path = 'dataset/validated.csv'  # Path to the metadata CSV
    audio_base_dir = 'dataset/pv/'  # Base directory containing audio files

    # Example: Separate real and fake audio directories based on metadata
    real_audio_dir = os.path.join(audio_base_dir, 'real_audio/')
    fake_audio_dir = os.path.join(audio_base_dir, 'fake_audio/')
    output_img_dir = 'processed_spectrograms/'  # Directory to save spectrograms

    # Ensure directories exist
    os.makedirs(real_audio_dir, exist_ok=True)
    os.makedirs(fake_audio_dir, exist_ok=True)

    print(f"Processing real audio directory: {real_audio_dir}")
    print(f"Processing fake audio directory: {fake_audio_dir}")
    if not os.listdir(real_audio_dir):
        raise ValueError(f"No files found in real audio directory: {real_audio_dir}")
    if not os.listdir(fake_audio_dir):
        raise ValueError(f"No files found in fake audio directory: {fake_audio_dir}")

    # Train model or load existing model
    if not os.path.exists('deepfake_audio_classifier_rf.pkl'):
        model = train_model(real_audio_dir, fake_audio_dir, output_img_dir)
    else:
        model = joblib.load('deepfake_audio_classifier_rf.pkl')
        print("Model loaded successfully!")

    # Example: Predict if a given audio file is real or fake
    test_audio_file = r"E:\voice file\dataset\pv\test_audio\Hareesh Audio.wav"
    predict_audio(test_audio_file, model)
