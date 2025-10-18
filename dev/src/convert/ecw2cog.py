from sys import argv
import subprocess
import glob

def run(src_filepath, dst_filepath):
    #list file names of folder (mypath = folder name)
    import os
    from os import listdir
    print(os.getcwd())
    print(src_filepath)
    print(os.path.isfile(str(src_filepath)))
    src = os.path.basename(str(src_filepath))
    from os.path import isfile, join

    if not os.path.exists("img/compliant-cog/"):
        os.makedirs("img/compliant-cog/")

    #translate to GeoTiff using src and dst
    print("Starting conversion of {}".format(src))
    subprocess.call(["gdal_translate", "-of", "GTiff", str(src_filepath), str(dst_filepath), "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS","--config", "GDAL_CACHEMAX","512"])
    print("{} successfully converted".format(src))
    subprocess.call(["gdaladdo", str(dst_filepath)])
    subprocess.call(["gdal_translate", "-of", "GTiff", dst_filepath, str(os.path.basename(dst_filepath).split(".")[0]+"-compliant.tif"), "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "BIGTIFF=YES", "-co","COPY_SRC_OVERVIEWS=YES", "-co","NUM_THREADS=ALL_CPUS","--config", "GDAL_CACHEMAX","512"])


if __name__ == '__main__':
    run(argv[1], argv[2])
