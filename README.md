
# Aos telhados brasis.

The dataset is a drone (.tif) file with 3cm of resolution from a neighborhood in Brasil. 

## Sync uv venv

```bashs
uv venv
source .venv/bin/activate 
uv sync
```
## How to read the repo

### Visualize the drone file.
to visualize the tif file.

```bash
rio viz /mnt/d/Desktop/drone-mapping/data/cog_3857.tif
```
It opens a web-broweser. The COG should be in 3857

## Steps

### 1. Create the patches of the original drone image (512x512).

To do it I am using the [cog2tile](https://github.com/kshitijrajsharma/cog2tiles) library. The folder old contains an implementation by hand on the `old_notebook/patchify.py` however is not taking in consideration the Tile Map Service.  Due this reason, for reconstructing the patches back to the original image, cog2tiles was the choice. 

    ### Convert COG into tile
    ``bash
    cog2tiles /mnt/d/desktop/drone-mapping/data/cog_3857.tif -z 19 --tile-size 512 --extension tif --output-dir /mnt/d/desktop/drone-mapping/data/new_tiling/tiles/ 
    ```

The zoom level 19 contains a resolution sparser than 3cm. However, the highest zoom level dont fit the entire rooftops in one tile, most of them are cutted in half. So 19 was an arbitrary choice. How the zoom levels afect the dowstream tasks is something to be explored. 

### 2. Annotate the original image.

The annotation was done with QGIS. A shapefile was created with the bounding boxes of each rooftop. It was annotated 


Visualize the annotation overlaying the original image
```bash
viewtif /media/manecao/HD/Desktop/drone-mapping/data/cog3f_3857.tif --shapefile /media/manecao/HD/Desktop/drone-mapping/data/shp/building-footprint.shp --scale 10 --rgb 1 2 3 
```

### 3. Patchify the shapefile and convert to Binary masks. 

The `dev.patchify` notebook contains the steps to patchify the shapefile and convert each patch into a binary mask.
To run it properly it is necessary to config the (yaml)["./dev/src/config/config.yaml"] file indicating the paths. 

The given notebook also set up the dataset into the yolo format. 

 The binary masks are saved in a folder structure with the following structure:

```
├── new_tiling
│   ├── bitmap ## binary masks
│   ├── labels ## labels to train yolo
│   ├── labels_segmentation ## labels to train segmentation models
│   ├── tiles ## original tiles
│   ├── tiles_metadata ## metadata of each tile
│   └── YOLO ## yolo formatted data
│       ├── images
│       │   ├── test
│       │   └── train
│       └── labels
│           ├── test
│           └── train
```

4 . Training the model

The file `dev/trainingyolo.ipynb` contains the training experimental part. Whereas the `config.yaml` contains the Yolo configuration.
The data for the training is inside the `dev/data/zip`folder. 





