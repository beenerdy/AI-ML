import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def generate_hash(file_name):
    """Generate a unique hash for a file based on its name."""
    hasher = hashlib.md5()
    hasher.update(file_name.encode("utf-8"))
    return hasher.hexdigest()


# Parse XML to dictionary
def parse_dict(element):
    result = {}
    iterator = iter(element)
    for child in iterator:
        if child.tag == "key":
            key = child.text
            try:
                value = next(iterator)
                if value.tag == "integer":
                    result[key] = (
                        int(value.text) if "." not in value.text else float(value.text)
                    )
                elif value.tag == "string":
                    result[key] = value.text
                elif value.tag == "true":
                    result[key] = True
                elif value.tag == "false":
                    result[key] = False
                elif value.tag == "dict":
                    result[key] = parse_dict(value)
                elif value.tag == "array":
                    result[key] = parse_array(value)
            except StopIteration:
                break
    return result


# Handle arrays in XML parsing
def parse_array(element):
    result = []
    for child in element:
        if child.tag == "dict":
            result.append(parse_dict(child))
        elif child.tag == "array":
            result.append(parse_array(child))
        elif child.tag == "integer":
            result.append(int(child.text))
        elif child.tag == "string":
            result.append(child.text)
    return result


# Process the input dictionary to extract playlists and tracks
def process_input(input_object):
    playlists = []
    for playlist_data in input_object.get("Playlists", []):
        playlist = {
            "Name": playlist_data.get("Name"),
            "Description": playlist_data.get("Description"),
            "Playlist ID": playlist_data.get("Playlist ID"),
            "Playlist Persistent ID": playlist_data.get("Playlist Persistent ID"),
            "Playlist Items": [],
        }
        for playlist_item in playlist_data.get("Playlist Items", []):
            track_id = playlist_item.get("Track ID")
            if str(int(track_id)) in input_object.get("Tracks", {}):
                track_data = input_object["Tracks"][str(int(track_id))]
                playlist_item.update(track_data)
                playlist["Playlist Items"].append(playlist_item)
        playlists.append(playlist)
    return playlists


def parse_and_extract_xml(file_path, start_index=0):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the first 'dict' tag, since the root may not be 'dict'
    tracks_dict = root.find("dict")
    if tracks_dict is None:
        raise ValueError("No 'dict' tag found in the XML structure.")

    # Convert the XML structure to a dictionary using the existing parse_dict function
    itunes_library = parse_dict(tracks_dict)

    # Process the dictionary to extract playlists and tracks using the existing process_input function
    playlists = process_input(itunes_library)

    # Initialize the song paths and json_data
    song_paths = []
    json_data = {"Playlist Name": "Imported Playlist", "Playlist Items": []}

    # Extract track details and paths
    for playlist in playlists:
        for track in playlist["Playlist Items"]:
            location = track.get("Location")
            if location:
                # Decode the URL-encoded path and remove 'file://'
                song_path = unquote(location.replace("file://", ""))
                song_paths.append(song_path)

                # Add track details to json_data
                json_data["Playlist Items"].append(track)

    # Return the paths starting from the specified index and the constructed json_data
    return song_paths[start_index:], json_data


def extract_features(
    features_cache, file_name, check_cache=True, num_segments=4, segment_duration=20
):
    """
    Extract various audio features from a given audio file by sampling multiple segments.

    Args:
        file_name (str): The path to the audio file.
        num_segments (int): Number of segments to sample from the song.
        segment_duration (int): Duration of each segment in seconds.

    Returns:
        dict: A dictionary of extracted audio features if successful, None otherwise.
    """
    # Generate a unique key for the file
    file_hash = generate_hash(file_name)

    # Define the required features
    required_features = [
        "mfccs_mean",
        "chroma_mean",
        "spectral_contrast_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "rms_mean",
        "harmonic_mean",
        "percussive_mean",
        "spectral_rolloff_mean",
        "tonnetz_mean",
        "spectral_bandwidth_mean",
        "spectral_flatness_mean",
        "tempo_mean",
        "onset_strength_mean",
    ]

    # Check if features are already cached
    if file_hash in features_cache and check_cache:
        cached_features = features_cache[file_hash]
        if all(feature in cached_features for feature in required_features):
            return cached_features

    # If any features are missing, proceed with the extraction
    cached_features = {}

    try:
        # Load audio to get the total duration
        y, sr = librosa.load(file_name, sr=None)
        total_duration = len(y) / sr

        # Ensure the song is long enough to extract the desired segments
        if total_duration < segment_duration * num_segments:
            print(f"Audio too short: {file_name}")
            return None

        # Calculate the starting points for each segment
        segment_starts = [
            0,  # Start of the song
            max(
                0, (total_duration / 3) - (segment_duration / 2)
            ),  # Start of the middle segment
            max(
                0, (2 * total_duration / 3) - (segment_duration / 2)
            ),  # Start of the second middle segment
            max(0, total_duration - segment_duration),  # End of the song
        ]

        # Initialize feature accumulators only if they are not in the cache
        mfccs_accum = [] if "mfccs_mean" not in cached_features else None
        chroma_accum = [] if "chroma_mean" not in cached_features else None
        spectral_contrast_accum = (
            [] if "spectral_contrast_mean" not in cached_features else None
        )
        zcr_accum = [] if "zcr_mean" not in cached_features else None
        spectral_centroid_accum = (
            [] if "spectral_centroid_mean" not in cached_features else None
        )
        rms_accum = [] if "rms_mean" not in cached_features else None
        harmonic_accum = [] if "harmonic_mean" not in cached_features else None
        percussive_accum = [] if "percussive_mean" not in cached_features else None
        spectral_rolloff_accum = (
            [] if "spectral_rolloff_mean" not in cached_features else None
        )
        tonnetz_accum = [] if "tonnetz_mean" not in cached_features else None
        spectral_bandwidth_accum = (
            [] if "spectral_bandwidth_mean" not in cached_features else None
        )
        spectral_flatness_accum = (
            [] if "spectral_flatness_mean" not in cached_features else None
        )
        tempo_accum = [] if "tempo_mean" not in cached_features else None
        onset_strength_accum = (
            [] if "onset_strength_mean" not in cached_features else None
        )

        for start in segment_starts:
            y_segment = y[int(sr * start) : int(sr * (start + segment_duration))]

            # MFCCs
            if mfccs_accum is not None:
                mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
                mfccs_accum.append(np.mean(mfccs.T, axis=0))

            # Chroma features
            if chroma_accum is not None:
                chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
                chroma_accum.append(np.mean(chroma.T, axis=0))

            # Spectral contrast
            if spectral_contrast_accum is not None:
                spectral_contrast = librosa.feature.spectral_contrast(
                    y=y_segment, sr=sr
                )
                spectral_contrast_accum.append(np.mean(spectral_contrast.T, axis=0))

            # Zero-Crossing Rate
            if zcr_accum is not None:
                zcr = librosa.feature.zero_crossing_rate(y_segment)
                zcr_accum.append(np.mean(zcr))

            # Spectral Centroid
            if spectral_centroid_accum is not None:
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=y_segment, sr=sr
                )
                spectral_centroid_accum.append(np.mean(spectral_centroid))

            # RMS (Root Mean Square Energy)
            if rms_accum is not None:
                rms = librosa.feature.rms(y=y_segment)
                rms_accum.append(np.mean(rms))

            # Harmonic/Percussive Separation
            if harmonic_accum is not None:
                harmonic, percussive = librosa.effects.hpss(y_segment)
                harmonic_accum.append(np.mean(harmonic))
                percussive_accum.append(np.mean(percussive))

            # Spectral Roll-off
            if spectral_rolloff_accum is not None:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
                spectral_rolloff_accum.append(np.mean(spectral_rolloff))

            # Tonnetz
            if tonnetz_accum is not None:
                tonnetz = librosa.feature.tonnetz(y=y_segment, sr=sr)
                tonnetz_accum.append(np.mean(tonnetz.T, axis=0))

            # Spectral Bandwidth
            if spectral_bandwidth_accum is not None:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(
                    y=y_segment, sr=sr
                )
                spectral_bandwidth_accum.append(np.mean(spectral_bandwidth))

            # Spectral Flatness
            if spectral_flatness_accum is not None:
                spectral_flatness = librosa.feature.spectral_flatness(y=y_segment)
                spectral_flatness_accum.append(np.mean(spectral_flatness))

            # Tempo and Onset Strength
            if tempo_accum is not None or onset_strength_accum is not None:
                onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
                if tempo_accum is not None:
                    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                    tempo_accum.append(tempo)
                if onset_strength_accum is not None:
                    onset_strength_accum.append(np.mean(onset_env))

        # Aggregate all features into the cache if they were calculated
        if mfccs_accum is not None:
            cached_features["mfccs_mean"] = np.mean(mfccs_accum, axis=0).tolist()
        if chroma_accum is not None:
            cached_features["chroma_mean"] = np.mean(chroma_accum, axis=0).tolist()
        if spectral_contrast_accum is not None:
            cached_features["spectral_contrast_mean"] = np.mean(
                spectral_contrast_accum, axis=0
            ).tolist()
        if zcr_accum is not None:
            cached_features["zcr_mean"] = float(np.mean(zcr_accum))
        if spectral_centroid_accum is not None:
            cached_features["spectral_centroid_mean"] = float(
                np.mean(spectral_centroid_accum)
            )
        if rms_accum is not None:
            cached_features["rms_mean"] = float(np.mean(rms_accum))
        if harmonic_accum is not None:
            cached_features["harmonic_mean"] = float(np.mean(harmonic_accum))
        if percussive_accum is not None:
            cached_features["percussive_mean"] = float(np.mean(percussive_accum))
        if spectral_rolloff_accum is not None:
            cached_features["spectral_rolloff_mean"] = float(
                np.mean(spectral_rolloff_accum)
            )
        if tonnetz_accum is not None:
            cached_features["tonnetz_mean"] = np.mean(tonnetz_accum, axis=0).tolist()
        if spectral_bandwidth_accum is not None:
            cached_features["spectral_bandwidth_mean"] = float(
                np.mean(spectral_bandwidth_accum)
            )
        if spectral_flatness_accum is not None:
            cached_features["spectral_flatness_mean"] = float(
                np.mean(spectral_flatness_accum)
            )
        if tempo_accum is not None:
            cached_features["tempo_mean"] = float(np.mean(tempo_accum))
        if onset_strength_accum is not None:
            cached_features["onset_strength_mean"] = float(
                np.mean(onset_strength_accum)
            )

        if check_cache:
            # Cache the features
            features_cache[file_hash] = cached_features
            # Save the cache after each song is processed
            save_cache(features_cache)

        return cached_features

    except Exception as e:
        print(f"Error encountered while parsing file: {file_name} - error: {e}")
        return None


def preprocess_feature(feature, method):
    if method == "log1p":
        return np.log1p(feature)
    elif method == "normalize":
        scaler = MinMaxScaler()
        return scaler.fit_transform(feature.reshape(-1, 1)).flatten()
    elif method == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(feature.reshape(-1, 1)).flatten()
    return feature


def separate_and_process_features(entry, pca_components=None):
    # Access the key and the corresponding feature dictionary
    hash_key = list(entry.keys())[0]
    features = entry[
        hash_key
    ]  # Access the first element of the tuple which is the features dictionary

    # Define the preprocessing methods for each feature
    preprocessing_methods = {
        "mfccs_mean": "standardize",
        "chroma_mean": "normalize",
        "spectral_contrast_mean": "normalize",
        "tonnetz_mean": "normalize",
        "zcr_mean": None,  # No preprocessing for this feature
        "spectral_centroid_mean": "standardize",
        "rms_mean": "log1p",
        "harmonic_mean": "standardize",
        "percussive_mean": "standardize",
        "spectral_rolloff_mean": "log1p",
        "spectral_bandwidth_mean": "standardize",
        "spectral_flatness_mean": "normalize",
        "tempo_mean": None,  # No preprocessing for this feature
        "onset_strength_mean": "normalize",
    }

    # Prepare lists for processed time-series and static features
    time_series_features_list = []
    static_features_list = []

    # Feature sets
    time_series_keys = [
        "mfccs_mean",
        "chroma_mean",
        "spectral_contrast_mean",
        "tonnetz_mean",
    ]
    static_keys = [
        "zcr_mean",
        "spectral_centroid_mean",
        "rms_mean",
        "harmonic_mean",
        "percussive_mean",
        "spectral_flatness_mean",
        "tempo_mean",
        "onset_strength_mean",
    ]

    # Process time-series features
    for key in time_series_keys:
        if key in features:  # Check if the feature exists
            feature = np.array(features[key])
            method = preprocessing_methods.get(key)
            processed_feature = preprocess_feature(feature, method)
            time_series_features_list.append(processed_feature)

    # Process static features
    for key in static_keys:
        if key in features:  # Check if the feature exists
            feature = np.array(features[key]).reshape(-1, 1)
            method = preprocessing_methods.get(key)
            processed_feature = preprocess_feature(feature, method)
            static_features_list.append(processed_feature.flatten())

    # Concatenate processed time-series features
    time_series_features = (
        np.concatenate(time_series_features_list)
        if time_series_features_list
        else np.array([])
    )

    # Apply PCA (optional) to time-series features
    if pca_components and time_series_features.size > 0:
        pca = PCA(n_components=pca_components)
        time_series_features = pca.fit_transform(
            time_series_features.reshape(1, -1)
        ).flatten()

    # Concatenate processed static features
    static_features = (
        np.concatenate(static_features_list) if static_features_list else np.array([])
    )

    return time_series_features, static_features
