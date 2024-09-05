# Music Categorization and Playlist Creation using Machine Learning

This project involves analyzing and categorizing music tracks based on user preferences using machine learning techniques. By extracting various audio features and employing deep learning models, it classifies tracks into "liked" or "disliked" categories, generating personalized playlists compatible with Apple Music, which manages local files.

## Project Overview

The goal of this project is to build a system that classifies music tracks based on audio features and generates playlists for users. The system processes each track by extracting features like MFCCs, chroma, spectral contrast, etc., and uses a combination of time-series and static feature classification to determine the user’s preference.

### Key Features

- **MFCCs, Chroma, Spectral Contrast**: Extracted using `librosa` and `essentia` to represent tonal, harmonic, and rhythmic features.
- **Data Preprocessing**: Time-series and static features are separated, normalized, and standardized.
- **Modeling**: A custom neural network with Conv1D layers, LSTM units, and attention mechanisms processes time-series features, while Dense layers handle static features.
- **Playlist Generation**: Based on the classification results, playlists are generated in XML format, compatible with Apple Music, which manages local music files.

## Data Preprocessing

1. **Feature Extraction**:
   - Extracted multiple features such as MFCCs, chroma, spectral contrast, ZCR, spectral centroid, and RMS energy from audio files to capture key audio characteristics.
   - Sampled 4 segments from each track to ensure comprehensive coverage of each song.

2. **Data Handling**:
   - **Caching**: Each track's features were cached after extraction to avoid redundant calculations and improve efficiency.
   - **Balancing**: Used upsampling and downsampling techniques to handle imbalanced classes of "liked" and "disliked" songs, ensuring equal representation during model training.

3. **Static vs Time-Series Features**:
   - Time-series features (MFCCs, chroma, spectral contrast, tonnetz) were stored in one structure, while static features (ZCR, spectral centroid, RMS, harmonic, percussive components, etc.) were stored in another.
   - The static and time-series features were normalized and standardized to ensure they were on similar scales for better model performance:
     - **Normalization**: Used `MinMaxScaler` to normalize certain static features (e.g., chroma, spectral contrast, tonnetz) to the range [0, 1].
     - **Standardization**: Applied `StandardScaler` to time-series features like MFCCs and spectral centroid to normalize them to a standard distribution.
     - **Log Transformation**: Performed log transformations on features like RMS and spectral rolloff to reduce skewness.

4. **Outlier Removal**:
   - Removed outliers using z-score and IQR (Interquartile Range) methods to ensure that extreme values didn’t distort the model's learning process.

5. **PCA for Visualization**:
   - **Principal Component Analysis (PCA)** was applied solely for display purposes to visualize the feature space. It was not used for reducing dimensionality in the model training.

## Requirements

- Python 3.x
- Libraries: `librosa`, `essentia`, `scikit-learn`, `tensorflow`, `pandas`
  
Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd <repo-folder>
    ```
2. Run the Jupyter Notebook:
    ```bash
    jupyter notebook "Music Sorting with ML.ipynb"
    ```

3. Prepare your music files and metadata in XML format. This will serve as input for the classification and playlist generation process.

## Model Training and Evaluation

The model consists of a combination of convolutional layers and LSTMs for time-series analysis and dense layers for static features. The combined outputs are classified into "liked" or "disliked," and playlists are generated accordingly. The model is trained and evaluated using accuracy and ROC AUC metrics.

### Key Model Details:
- **Time-series processing**: Conv1D and bidirectional LSTM layers extract temporal patterns from features like MFCCs and chroma.
- **Attention Mechanism**: Multi-head attention was applied to improve the model’s focus on important sections of each track.
- **Static Feature Processing**: Dense layers were used for static features, with batch normalization and dropout to prevent overfitting.

## Playlist Generation

After classifying the tracks, playlists are generated in XML format compatible with **Apple Music**:
- `liked_songs_playlist.xml`
- `disliked_songs_playlist.xml`
- `semi-liked_songs_playlist.xml`

These playlists can be imported into Apple Music, which manages local files.

## Results and Interpretation

The classification model achieves a good balance between accuracy and interpretability, making it suitable for personalized playlist creation.
