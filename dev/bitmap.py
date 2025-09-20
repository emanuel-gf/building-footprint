import os
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from PIL import Image
import logging

def generate_bitmaps_for_patches(intersection_gdf, output_folder, 
                                    bitmap_size=(512, 512), 
                                    fill_value=0, 
                                    dtype=np.uint8,
                                    save_format='PNG'):
    """
    Generate and save bitmaps for each unique patch in the intersection GeoDataFrame.
    The GeoDataFrame is the spatial join between a Building Footprint and gdf of patches and its metadata.
    
    The generator uses the unique list of patches that were intersected/overlap in the spatial join. 
    For each path, it finds the geometries of building footprints and create a bitmap saving with the regarding file name. 
    
    Parameters:
    -----------
    intersection_gdf : GeoDataFrame
        Input GeoDataFrame containing patch data with geometries and transforms
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
    
    # Define required columns
    cols_affine = ['transform_a', 'transform_b', 'transform_c', 
                   'transform_d', 'transform_e', 'transform_f']
    
    # Validate input
    required_cols = ['patch_id', 'filename', 'geometry'] + cols_affine
    missing_cols = [col for col in required_cols if col not in intersection_gdf.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize result tracking
    results = {'success': [], 'errors': []}
    
    # Process each patch
    unique_patches = sorted(intersection_gdf["patch_id"].unique())
    total_patches = len(unique_patches)
    
    for i, patch_id in enumerate(unique_patches, 1):
        print(f"Processing patch {patch_id} ({i}/{total_patches})")
        
        try:
            # Get patch data
            patch_data = intersection_gdf[intersection_gdf["patch_id"] == patch_id]
            
            # Validate single filename per patch
            unique_filenames = patch_data["filename"].unique()
            if len(unique_filenames) != 1:
                raise ValueError(f"Patch {patch_id} has {len(unique_filenames)} filenames, expected 1")
            
            filename = unique_filenames[0]
            
            # Prepare geometries for rasterization
            geometries = [(geom, 1) for geom in patch_data["geometry"] if geom is not None]
            
            if not geometries:
                raise ValueError(f"No valid geometries found for patch {patch_id}")
            
            # Get affine transform parameters
            transform_params = patch_data[cols_affine].iloc[0].values
            
            # Validate transform parameters
            if np.any(np.isnan(transform_params)):
                raise ValueError(f"NaN values found in transform parameters for patch {patch_id}")
            
            transform = Affine(*transform_params)
            
            # Generate bitmap
            bitmap = rasterize(
                geometries,
                out_shape=bitmap_size,
                transform=transform,
                fill=fill_value,
                dtype=dtype
            )
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_filename = f"{base_name}.{save_format.lower()}"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save bitmap
            save_bitmap(bitmap, output_path, save_format)
            
            results['success'].append(patch_id)
            print(f"  ✓ Saved: {output_filename}")
            
        except Exception as e:
            error_msg = f"Error processing patch {patch_id}: {str(e)}"
            print(f"  ✗ {error_msg}")
            logging.error(error_msg)
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