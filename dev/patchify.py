import rasterio
from rasterio.windows import Window
from pathlib import Path
import json
import os
import glob
import geopandas as gpd
from shapely.geometry import box, mapping
from rasterio.transform import Affine
from rasterio import features
from shapely.affinity import affine_transform

def create_and_save_patches(cog_path,
                            output_dir,
                            patch_size=512,
                            overlap=64, 
                            bands=None,
                            save_format='jpg',
                            verbose=True):
    """Create and save fixed-size patches from a Cloud-Optimized GeoTIFF (COG).

    This function reads a raster (typically a COG), extracts square patches
    with a specified size and overlap, saves each patch to disk (either as
    JPEG images or GeoTIFFs) and writes a JSON metadata file describing
    each patch. Crucially, the metadata stores the patch-local affine
    transform (computed using :func:`rasterio.windows.transform`) so the
    spatial location of pixels inside each patch can be recovered and used
    for rasterization or vectorization workflows.

    Parameters
    ----------
    cog_path : str or pathlib.Path
        Path to the input Cloud-Optimized GeoTIFF (or any readable raster).
    output_dir : str or pathlib.Path
        Directory where `patches/` and `metadata/` subdirectories will be
        created. Patches and the metadata JSON are written under these
        folders.
    patch_size : int, optional
        Size (in pixels) of the square patches to extract. Default is 512.
    overlap : int, optional
        Number of pixels of overlap between adjacent patches. Default is 64.
    bands : sequence of int, optional
        Sequence of 1-based band indices to read from the source raster.
        If ``None``, all bands are read. Example: ``[1,2,3]`` for RGB.
    save_format : {'jpg', 'tif'} or str, optional
        Output format for saved patches. Use ``'jpg'`` to save simple
        image files (no georeferencing written) or ``'tif'`` (default
        recommended) to save GeoTIFFs that preserve CRS and the patch-local
        affine transform. The check is case-insensitive.
    verbose : bool, optional
        If True, print progress and a small debug dump for the first
        produced patch (useful to validate the patch-local transform).

    Returns
    -------
    list of dict
        A list of metadata dictionaries, one per patch. Each dictionary
        contains keys including ``patch_id``, ``filename``, ``original_window``,
        ``transform`` (the 6-element affine transform for the patch in
        rasterio's Affine order), ``crs``, ``bands``, ``original_file``,
        and ``original_transform``.

    Notes
    -----
    - When ``save_format='jpg'``, patches are saved as normal images and
      no geospatial tags are included in the files; the metadata JSON
      still contains the patch-local transform and CRS string so spatial
      reprojection and rasterization remain possible.
    - The stored ``transform`` is suitable for use with rasterio's
      ``transform * (col, row)`` to find geographic coordinates for a
      pixel in the patch, and ``~Affine`` (the inverse) to go back to
      patch pixel coordinates.
    - The function iterates patches by stepping with ``patch_size - overlap``
      so the last columns/rows are only created where a full patch fits.
      If you want to include partial edge patches, modify the loop logic
      to pad and include remainder windows.

    Example
    -------
    >>> create_and_save_patches('input.tif', 'output_dir', patch_size=256,
    ...                         overlap=32, bands=[1,2,3], save_format='tif')


    """
    output_dir = Path(output_dir)
    patches_dir = output_dir / 'patches'
    metadata_dir = output_dir / 'metadata'
    
    patches_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    patch_metadata = []
    patch_id = 0
    
    with rasterio.open(cog_path) as src:
        print(f"Original image: {src.width}x{src.height}, {src.count} bands")
        print(f"Original transform: {src.transform}")
        
        # Read specific bands or all
        if bands is None:
            bands = list(range(1, src.count + 1))
        
        height, width = src.height, src.width
        
        for row in range(0, height - patch_size + 1, patch_size - overlap):
            for col in range(0, width - patch_size + 1, patch_size - overlap):
                window = Window(col, row, patch_size, patch_size)
                
                # Read patch data
                patch_data = src.read(bands, window=window)
                
                # rasterio.windows.transform gives patch-local transform
                patch_transform = rasterio.windows.transform(window, src.transform)
                
                if verbose:
                    # Let's verify this is correct by testing corner coordinates
                    if patch_id == 0:  # Debug first patch
                        print(f"\nDEBUG - First patch (ID {patch_id}):")
                        print(f"  Window: {window}")
                        print(f"  Patch transform: {patch_transform}")
                        
                        # Test: patch pixel (0,0) should map to some geo coordinate
                        geo_x, geo_y = patch_transform * (0, 0)
                        print(f"  Patch pixel (0,0) -> geo({geo_x:.2f}, {geo_y:.2f})")
                        
                        # Test: patch pixel (patch_size, patch_size) should map to different geo coordinate  
                        geo_x2, geo_y2 = patch_transform * (patch_size, patch_size)
                        print(f"  Patch pixel ({patch_size},{patch_size}) -> geo({geo_x2:.2f}, {geo_y2:.2f})")
                        
                        # Verify inverse works
                        inv_transform = ~patch_transform
                        px, py = inv_transform * (geo_x, geo_y)
                        print(f"  Inverse check: geo({geo_x:.2f}, {geo_y:.2f}) -> pixel({px:.2f}, {py:.2f})")

                
                # Save patch - you can save as JPG if you want, but GeoTIFF preserves spatial info better
                if save_format.lower() == 'jpg':
                    patch_filename = f"patch_{patch_id:06d}.jpg"
                    patch_path = patches_dir / patch_filename
                    
                    # For JPG, save without spatial info (just the image)
                    from PIL import Image
                    import numpy as np
                    
                    # Convert to uint8 if needed and transpose for PIL (height, width, channels)
                    if patch_data.dtype != np.uint8:
                        # Normalize to 0-255 range
                        patch_data_norm = ((patch_data - patch_data.min()) / 
                                         (patch_data.max() - patch_data.min()) * 255).astype(np.uint8)
                    else:
                        patch_data_norm = patch_data
                    
                    # Transpose from (bands, height, width) to (height, width, bands)
                    if len(bands) == 1:
                        img_array = patch_data_norm[0]  # Grayscale
                    else:
                        img_array = np.transpose(patch_data_norm, (1, 2, 0))  # RGB
                    
                    Image.fromarray(img_array).save(patch_path)
                    
                else:
                    # Save as GeoTIFF (recommended)
                    patch_filename = f"patch_{patch_id:06d}.tif"
                    patch_path = patches_dir / patch_filename
                    
                    with rasterio.open(
                        patch_path, 'w',
                        driver='GTiff',
                        height=patch_size,
                        width=patch_size,
                        count=len(bands),
                        dtype=patch_data.dtype,
                        crs=src.crs,
                        transform=patch_transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(patch_data)
                
                # Store metadata with the CORRECT patch-local transform
                metadata = {
                    'patch_id': patch_id,
                    'filename': patch_filename,
                    'original_window': {
                        'col_off': window.col_off,
                        'row_off': window.row_off,
                        'width': window.width,
                        'height': window.height
                    },
                    'transform': list(patch_transform)[:6],  # Patch-local affine transform
                    'crs': str(src.crs),
                    'bands': bands,
                    'original_file': str(cog_path),
                    'original_transform': list(src.transform)[:6],  # Store original for reference
                }
                
                patch_metadata.append(metadata)
                patch_id += 1
                
                if patch_id % 100 == 0:
                    print(f"Created {patch_id} patches...")
    
    # Save metadata
    metadata_file = metadata_dir / 'patches_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(patch_metadata, f, indent=2)
    
    print(f"Created {patch_id} patches in {patches_dir}")
    print(f"Metadata saved to {metadata_file}")
    
    
    return patch_metadata




def get_transform(patch):
    """
    Return affine object from the metadata transform
    """
    return Affine(patch['transform'][0], patch['transform'][1], patch['transform'][2],
                      patch['transform'][3], patch['transform'][4], patch['transform'][5])
    
    
def get_patch_bounds_from_metadata(patch):
    """
    Create the bbox of the patch in pixel and geographic coordinates
    """
    ## affine
    transform = get_transform(patch)
    
    # Get pixel dimensions
    width = patch['original_window']['width']
    height = patch['original_window']['height']
    col_off = patch['original_window']['col_off']
    row_off = patch['original_window']['row_off']
    
    ## pixel bounds
    bbox = box(col_off, row_off, col_off+width, row_off+height)
    
    ## geographic bounds
    bbox_transform = affine_transform(bbox, transform.to_shapely())
    
    return bbox_transform, bbox  # xmin, ymin, xmax, ymax