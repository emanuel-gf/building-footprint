import yaml
import os
import json 
from pathlib import Path

def load_config(config_path="cfg/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_json(path_name):
    with open(path_name, 'r') as f:
        return json.load(f)


def create_list_files(folder, glob=None):
    """
    args:
        folder: Path
            Full-path of the folder holding the files to be listed
        glob: list of str
            Which patterns to glob (e.g., ["*.jpg", "*.shp", "*.txt"])
    """
    if not glob:
        glob = ["*.tif", "*.jpg", "*.png", "*.txt", "*.shp"]
    else:
        if not isinstance(glob, list):
            glob = [glob] 
    
    folder_path = Path(folder)
    files = [f for pattern in glob for f in folder_path.glob(pattern)]
    
    return files


def compose_zip_folder(list_folder1, list_folder2):
    """
    Compose two list of files into a matching zip tuple.
    Example: List of image and list of labels, returns a list where each tuples containing the match between both files.
    The second folder should be the folder to be look at, so each file of the folder1 is being looked at the folder2
    
    args:
        list_folder1: List
            List of file paths inside a folder
    """
    # Create lookup dictionary
    lookup_dict = {Path(f).stem: f for f in list_folder1}

    # Find matches
    matching_pairs = [(tile, lookup_dict[Path(tile).stem]) 
                    for tile in list_folder2 
                    if Path(tile).stem in lookup_dict]
    return matching_pairs