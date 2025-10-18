import os
import geopandas as gpd
import cv2
from skimage.color import rgb2gray
import numpy as np 
import json

def find_tile_per_x_y_z(x:int,y:int,z:int):
    path = "tile-"+str(x)+"-"+str(y)+"-"+str(z)+".tif"
    return path

def add_folder_to_path_name(filename,folder):
    return os.path.join(folder,filename)


def map_path_to_tile_info(img_path):
    """
    Given an image path, it extracts the x, y, z tile information from the filename.
    The filename should contain the x_tile,y_tile and z_tile information within it. 
    """
    basename = os.path.basename(img_path)
    _, x, y, z =  os.path.splitext(basename)[0].split('-')
    return int(x), int(y), int(z) 


def create_metadata_patches(gdf, list_files,output_dir):
    """
    Create metadata in json for each tile. 
    Follow the same tile name structure to store it.
    """

    ## make dir 
    os.makedirs(output_dir, exist_ok=True)

    for num,tile in enumerate(list_files):
        
        ## Create a dir path
        basename = os.path.basename(tile)
        
        if num<10:
            print(f"Reading tile: {tile}")

        numx, numy, numz = map_path_to_tile_info(tile)
        
        ## Retrive the polygon information in the gdf
        loc_tile = (gdf["x_tile"]==numx) & (gdf["y_tile"]==numy)
        geometry = gdf.loc[loc_tile]["geometry"].iloc[0]
        id = gdf.loc[loc_tile]["id"].iloc[0]
        
        ## Add information about range of pixels color values to identify likely all white patches. 
        img = cv2.imread(tile, cv2.IMREAD_COLOR_RGB)
        gray = np.array(rgb2gray(img)*255).astype(np.uint8)
        min_ = int(gray.min())
        max_  = int(gray.max())
        diff = int(np.absolute(max_-min_))
        
        ## histogram
        bins = np.histogram(gray.flatten(), range=(0,255), bins=26)[0] 
        cumsum = bins.cumsum() / bins.sum()
        
        ## Calculate the percentage of white taking the second last bins in the cumulative histogram
        perc_of_white = np.round(1- cumsum[-2],4)
        
        ## take the first bin in the cumulative histogram
        perc_of_black =np.round(cumsum[0],4)
        
        ## sum 
        sum_b_w = perc_of_white+perc_of_black
        
        row_metadata = {
                'patch_id': id,
                "x_tile": numx,
                "y_tile": numy,
                "z_tile": numz,
                'filename': basename.split(".")[0],
                'crs': "EPSG:4326",
                'geometry':str(geometry),
                "max_gray_value":max_,
                "min_gray_value":min_,
                "max_min_diff": diff,
                "percentage_white": perc_of_white,
                'percentage_black': perc_of_black,
                "sum_black_white": sum_b_w
            }
        
        # Save metadata
        path_file = basename.split('.')[0]+".json"
        metadata_file = os.path.join(output_dir,path_file)
        
        with open(metadata_file, 'w') as f:
            json.dump(row_metadata, f, indent=2)
        
        print(f"Metadata saved to {path_file}")