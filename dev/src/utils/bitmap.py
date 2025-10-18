from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
from rasterio import features 
import numpy as np  
import os 

def create_binary_mask(polygons_gdf, bounds, resolution=256):
    """
    Create a binary mask from polygons within a bounding box.
    
    Parameters:
    -----------
    polygons_gdf : GeoDataFrame or GeoSeries
        The polygons to rasterize
    bounds_bbox :  # (minx, miny, maxx, maxy)
        The bounding box boundary
    resolution : int or tuple
        Output image size (height, width)
    
    Returns:
    --------
    mask : numpy array
        Binary mask where polygons = 1, background = 0
    transform : affine.Affine
        Geospatial transform for the raster
    """

    
    # Set resolution
    if isinstance(resolution, int):
        height = width = resolution
    else:
        height, width = resolution
    
    # Create transform
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                           width, height)
    
    # Prepare geometries for rasterization
    if isinstance(polygons_gdf, gpd.GeoDataFrame):
        geometries = polygons_gdf.geometry
    else:
        geometries = polygons_gdf
    
    # Create shapes list (geometry, value) tuples
    shapes = [(geom, 1) for geom in geometries if geom is not None]
    
    # Rasterize
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    return mask, transform


def generate_bitmaps(intersection_gdf,
                                 gdf_tiles_grid,
                                 output_folder, 
                                    bitmap_size=(512, 512), 
                                    fill_value=0, 
                                    dtype=np.uint8,
                                    save_format='PNG'):
    """
    Generate and save bitmaps for each unique patch in the intersection GeoDataFrame.
    The GeoDataFrame is the spatial join between a Building Footprint and gdf of patches.
    
    The generator uses the unique list of patches that were intersected/overlap in the spatial join. 
    For each path, it finds the geometries of building footprints and create a bitmap saving with the regarding file name. 
    
    Parameters:
    -----------
    intersection_gdf : GeoDataFrame
        Input GeoDataFrame containing patch data with geometries and transforms
    gdf_tiles_grid: GeoDataFrame
        Gdf of grids created from Mercantile
    output_folder : str
        Path to folder where bitmaps will be saved
    bitmap_size : tuple, default (512, 512)
        Size of output bitmap (height, width)
    fill_value : int, default 0
        Fill value for areas outside geometries
    dtype : numpy.dtype, default np.uint8
        Data type for the output bitmap
    save_format : str, default 'PNG'
        Image format for saving ('PNG', 'TIFF', 'JPEG', etc.)
    
    Returns:
    --------
    dict: Dictionary with 'success' and 'errors' keys containing lists of patch_ids
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize result tracking
    results = {'success': [], 'errors': []}
    
    ## column to look it up
    ## it should be a column referencing the patch ID, where each patch ID is unique
    col = "index_tile"
    # Process each patch
    unique_patches = sorted(intersection_gdf[col].unique())
    total_patches = len(unique_patches)
    
    for i, patch_id in enumerate(unique_patches):
        print(f"Processing Intersection {patch_id} ({i}/{total_patches})")
        patch_id=int(patch_id)
        try:
            # Get 
            patch_data = intersection_gdf.loc[intersection_gdf[col] == patch_id]
            
            # Retrive geoseries containing the building geometries
            geometries = patch_data["geometry"].values
        
            # Retrieve the boundary of the tile grid
            boundary_df = gdf_tiles_grid.loc[gdf_tiles_grid["id"]==patch_data["id_tile"].iloc[0]]["geometry"].bounds
            boundary_list = boundary_df.iloc[0].to_list()
            
            ## create the bitmap
            mask, transform = create_binary_mask(
                            geometries,  # or just the GeometryArray
                            boundary_list,
                            resolution=512 # adjust as needed
                        )
            
            ## save 
            filename = patch_data["path_name"].iloc[0]
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_filename = f"{base_name}.{save_format.lower()}"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save bitmap
            save_bitmap(mask, output_path, save_format)
            
            results['success'].append(patch_id)
            print(f"  ✓ Saved: {output_filename}")
            
        except Exception as e:
            error_msg = f"Error processing patch {patch_id}: {str(e)}"
            print(f"  ✗ {error_msg}")
            results['errors'].append(patch_id)
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"  Success: {len(results['success'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results['errors']:
        print(f"  Failed patches: {results['errors']}")
    
    return results


def save_bitmap(bitmap, output_path, save_format='PNG'):
    """
    Save bitmap array to file.
    
    Parameters:
    -----------
    bitmap : numpy.ndarray
        2D array representing the bitmap
    output_path : str
        Full path where to save the file
    save_format : str
        Image format ('PNG', 'TIFF', 'JPEG', etc.)
    """
    try:
        # Convert to PIL Image
        if bitmap.dtype != np.uint8:
            # Normalize to 0-255 range if not uint8
            if bitmap.max() > 1:
                bitmap_normalized = ((bitmap - bitmap.min()) / 
                                   (bitmap.max() - bitmap.min()) * 255).astype(np.uint8)
            else:
                bitmap_normalized = (bitmap * 255).astype(np.uint8)
        else:
            bitmap_normalized = bitmap
        
        # Create PIL Image
        image = Image.fromarray(bitmap_normalized)  # 'L' for grayscale
        
        # Save with appropriate format
        if save_format.upper() == 'TIFF':
            image.save(output_path, format='TIFF', compression='lzw')
        elif save_format.upper() == 'PNG':
            image.save(output_path, format='PNG', optimize=True)
        elif save_format.upper() == 'JPEG':
            image.save(output_path, format='JPEG', quality=95)
        else:
            image.save(output_path, format=save_format.upper())
            
    except Exception as e:
        raise Exception(f"Failed to save bitmap to {output_path}: {str(e)}")