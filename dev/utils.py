
import os 

def map_path_to_tile_info(img_path):
    """
    Given an image path, extract the x, y, z tile information from the filename.
    return x, y, z as intergers
    """
    basename = os.path.basename(img_path)
    _, x, y, z =  os.path.splitext(basename)[0].split('-')
    return int(x), int(y), int(z) 