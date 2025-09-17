path_image = "/mnt/d/desktop/drone-mapping/data/drone-tiff.tif"

import rasterio as rio 

with rio.open(path_image) as src:
    print(src.profile)
    print(src.crs)
    print(src.bounds)
    print(src.width, src.height)
    print(src.count)  # number of bands
    band1 = src.read(1)  # read the first band
    print(band1)  # numpy array of the first band
    print(band1.shape)