import argparse
import logging
import os
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import unquote
from xml.dom import minidom

import numpy as np
import pandas as pd
from mutagen import File
from tensorflow.keras.models import load_model
from tqdm import tqdm

from utilities import (extract_features, generate_hash, parse_and_extract_xml,
                       separate_and_process_features)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


# Define the argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process iTunes XML playlists")
    parser.add_argument(
        "--xml_file", type=str, required=True, help="Path to the iTunes XML file"
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=None,
        help="Maximum number of tracks to process",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index to begin processing tracks",
    )
    return parser.parse_args()


# Process new songs, predict their likelihood, and categorize into playlists
def process_new_songs(
    xml_file_path, features_cache, model, num_segments=4, max_tracks=None, start_index=0
):
    song_paths, json_data = parse_and_extract_xml(
        xml_file_path, start_index=start_index
    )

    # Adjust the list based on the max_tracks parameter
    if max_tracks is not None:
        song_paths = song_paths[:max_tracks]

    results = []
    logging.info(
        f"Processing {len(song_paths)} tracks starting from index {start_index}."
    )

    liked_songs, semiliked_songs, disliked_songs = [], [], []

    for song_path in tqdm(song_paths, desc="Processing new songs", unit="song"):
        if os.path.exists(song_path):
            file_hash = generate_hash(song_path)

            features = extract_features(features_cache, song_path, check_cache=False)
            if features is not None:
                features_cache[file_hash] = features

            if features is not None:
                time_series_features, static_features = separate_and_process_features(
                    {file_hash: features}
                )
                features_per_segment = time_series_features.shape[0] // num_segments
                time_series_reshaped = time_series_features[
                    : num_segments * features_per_segment
                ].reshape((1, num_segments, features_per_segment))
                static_features = np.array(static_features).reshape(1, -1)

                try:
                    prediction = model.predict([time_series_reshaped, static_features])[
                        0
                    ][0]
                    prediction_percentage = round(prediction * 100, 2)

                    # Add "File Path" and "Prediction" explicitly to the features dictionary
                    features["File Path"] = song_path
                    features["Prediction"] = float(prediction_percentage)
                    results.append(features)

                    if prediction_percentage > 75:
                        liked_songs.append(song_path)
                    elif 50 < prediction_percentage <= 75:
                        semiliked_songs.append(song_path)
                    else:
                        disliked_songs.append(song_path)

                    logging.info(f"Prediction: {prediction_percentage} - {song_path}.")
                except Exception as e:
                    logging.error(
                        f"Prediction error for {song_path}: {e}. Skipping song."
                    )
        else:
            logging.warning(f"File not found: {song_path}")

    logging.info("Processing completed.")

    # Convert results to a DataFrame and make sure "Prediction" and "File Path" are in the DataFrame
    if results and "Prediction" in results[0] and "File Path" in results[0]:
        df_results = pd.DataFrame(results)
    else:
        logging.error("No valid predictions were made, DataFrame is empty.")
        df_results = pd.DataFrame()

    return df_results, json_data


# Filter JSON data by paths
def filter_json_data_by_paths(play_list_name, json_data, song_paths):
    song_paths_set = set(song_paths)
    filtered_json_data = {"Playlist Name": play_list_name, "Playlist Items": []}

    for item in json_data["Playlist Items"]:
        if "Location" in item:
            decoded_path = unquote(item["Location"]).replace("file://", "")
            file_name = os.path.basename(decoded_path)
            if any(file_name in path for path in song_paths_set):
                filtered_json_data["Playlist Items"].append(item)

    return filtered_json_data


# Save filtered playlist as an XML file
def create_xml_playlist(json_data, output_file):
    plist = ET.Element("plist")
    plist.set("version", "1.0")
    dict_node = ET.SubElement(plist, "dict")

    # Add tracks
    tracks_node = ET.SubElement(dict_node, "key")
    tracks_node.text = "Tracks"
    tracks_dict_node = ET.SubElement(dict_node, "dict")

    for item in json_data["Playlist Items"]:
        track_id = item["Track ID"]
        key_node = ET.SubElement(tracks_dict_node, "key")
        key_node.text = str(track_id)
        track_dict_node = ET.SubElement(tracks_dict_node, "dict")

        for key, value in item.items():
            if key == "Track ID":
                continue
            key_node = ET.SubElement(track_dict_node, "key")
            key_node.text = key
            if isinstance(value, bool):
                ET.SubElement(track_dict_node, "true" if value else "false")
            elif isinstance(value, int):
                value_node = ET.SubElement(track_dict_node, "integer")
                value_node.text = str(value)
            elif isinstance(value, str):
                value_node = ET.SubElement(track_dict_node, "string")
                value_node.text = value

    # Convert the XML tree to a string and save it
    xml_string = ET.tostring(plist, encoding="utf-8", method="xml")
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="    ")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


# Example usage:
if __name__ == "__main__":
    # Parse arguments from the terminal
    args = parse_arguments()

    # Load the model and cache
    combined_model = load_model("combined_model.h5")
    features_cache = {}

    # Process songs based on passed arguments
    predictions_df, json_data = process_new_songs(
        args.xml_file,
        features_cache,
        combined_model,
        max_tracks=args.max_tracks,
        start_index=args.start_index,
    )

    liked_songs = predictions_df[predictions_df["Prediction"].astype(float) > 75][
        "File Path"
    ].tolist()
    semiliked_songs = predictions_df[
        (predictions_df["Prediction"].astype(float) <= 75)
        & (predictions_df["Prediction"].astype(float) > 50)
    ]["File Path"].tolist()
    disliked_songs = predictions_df[predictions_df["Prediction"].astype(float) <= 49][
        "File Path"
    ].tolist()

    liked_json_data = filter_json_data_by_paths("Liked Songs", json_data, liked_songs)
    semiliked_json_data = filter_json_data_by_paths(
        "Semi-liked Songs", json_data, semiliked_songs
    )
    disliked_json_data = filter_json_data_by_paths(
        "Disliked Songs", json_data, disliked_songs
    )

    create_xml_playlist(liked_json_data, "./data/pred_liked.xml")
    create_xml_playlist(semiliked_json_data, "./data/pred_semiliked.xml")
    create_xml_playlist(disliked_json_data, "./data/pred_disliked.xml")
