# Music Categorization and Playlist Creation using Machine Learning

This project focuses on categorizing music tracks based on user preferences using machine learning. It employs a custom neural network to classify songs as "liked," "semi-liked," or "disliked," and generates playlists in an XML format compatible with Apple Music.

## Project Overview

The system classifies music tracks by extracting audio features and processing them using a combination of time-series and static feature models. The resulting playlists are generated in XML format for seamless import into Apple Music, which manages local music files.

### Key Components

- **Audio Feature Extraction**: Extracts features like MFCCs, chroma, spectral contrast, and more using libraries such as `librosa` and `essentia`.
- **Modeling**: A neural network model combines convolutional layers, LSTMs, and attention mechanisms to process time-series features, while dense layers handle static features.
- **Playlist Generation**: Based on classification results, playlists are created and stored in XML files that are compatible with Apple Music.

## Data Preprocessing

1. **Feature Extraction**:
   - Features such as MFCCs, chroma, spectral contrast, ZCR, spectral centroid, and RMS energy are extracted from audio files using `librosa` and `essentia`.
   - Each track is divided into multiple segments to capture the full range of audio characteristics.

2. **Caching**:
   - Extracted features are cached to avoid redundant calculations during reprocessing.

3. **Data Balancing**:
   - Upsampling and downsampling methods are applied to ensure an even distribution of "liked" and "disliked" tracks during model training.

4. **Normalization & Standardization**:
   - Time-series and static features are normalized using `MinMaxScaler` and standardized using `StandardScaler` to ensure consistent scaling across features.

5. **Outlier Removal**:
   - Outliers are removed using z-score and IQR methods to avoid skewing the model's learning process.

## Model Training

The model architecture consists of convolutional layers, LSTM units, and attention mechanisms to handle time-series data. Static features are processed through dense layers with batch normalization and dropout to prevent overfitting.

### Key Model Details:
- **Time-series processing**: Conv1D and LSTM layers are used to capture temporal patterns in features like MFCCs and chroma.
- **Attention Mechanism**: Attention layers help the model focus on important sections of each track.
- **Static Feature Processing**: Dense layers handle static features, while batch normalization and dropout are used for regularization.

### Evaluation
The model is evaluated using accuracy and ROC AUC metrics.

## Running the Project

### Jupyter Notebook for Model Training

To train the model, run the Jupyter Notebook:
```bash
jupyter notebook "Music Sorting with ML.ipynb"
```
The notebook processes the dataset, trains the model, and saves it as `combined_model.h5`.

### Song Classifier Script

To classify new songs and generate playlists, run the `song_classifier.py` script:
```bash
python song_classifier.py --xml_file ./data/to_process.xml --max_tracks 1 --start_index 200
```

#### Key Parameters:
- `--xml_file`: The path to the XML file containing the song data to be processed.
- `--max_tracks`: (Optional) The maximum number of tracks to process. Default is all.
- `--start_index`: (Optional) The index from which to start processing tracks. Default is 0.

The classifier loads the pre-trained model from `combined_model.h5` and processes the songs, classifying them as "liked," "semi-liked," or "disliked."

## Utilities

The project includes a set of utility functions shared between the notebook and the classifier:
- `extract_features`: Extracts audio features for classification.
- `generate_hash`: Generates unique hashes for songs to ensure consistency across datasets.
- `parse_and_extract_xml`: Parses the iTunes XML file and extracts relevant metadata.
- `separate_and_process_features`: Separates and processes static and time-series features for model input.

## Playlist Generation

After classifying tracks, the script generates three playlists in XML format:
- `liked_songs_playlist.xml`
- `disliked_songs_playlist.xml`
- `semi-liked_songs_playlist.xml`

These playlists can be imported into Apple Music, where they will be available for playback based on user preferences.

## Results

The model performs well in categorizing tracks according to user preferences, striking a balance between accuracy and playlist personalization. The classification results can be interpreted easily, making the system effective for creating personalized playlists.
