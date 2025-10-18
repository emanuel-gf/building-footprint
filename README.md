

to visualize the cog file
rio viz /mnt/d/Desktop/drone-mapping/data/cog_3857.tif

It opens a web-broweser. The COG should be in 3857

## Convert COG into tile
cog2tiles /mnt/d/desktop/drone-mapping/data/cog_3857.tif -z 19 --tile-size 512 --extension tif --output-dir /mnt/d/desktop/drone-mapping/data/new_tiling/tiles/ 