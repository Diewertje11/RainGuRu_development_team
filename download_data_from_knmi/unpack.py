import os
import tarfile

download_dir = r"/DATAFOLDER/cluster_projects/pr/0949_10/reflectivity_tar_archive_20210408"
data_dir = r"/DATAFOLDER/cluster_projects/pr/0949_10"

def unpack_knmi_tar(filename):
    year = filename[27:31]
    month = filename[31:33]
    day = filename[33:35]
    foldername = f"{year}-{month}-{day}"
    
    year_dir = os.path.join(data_dir, year)
    if not os.path.isdir(year_dir):
        os.mkdir(year_dir)
        
    destination_path = os.path.join(year_dir, foldername)
    tar_path = os.path.join(download_dir, filename)
    if os.path.isdir(destination_path):
        print (filename, ' already unpacked')
    else:
        os.mkdir(destination_path)
        tar = tarfile.open(tar_path)
        tar.extractall(path=destination_path)
        print (filename, ' succesfully unpacked')

for e, filename in enumerate(os.listdir(download_dir)):
    if filename[-4:] == '.tar':
        unpack_knmi_tar(filename)
