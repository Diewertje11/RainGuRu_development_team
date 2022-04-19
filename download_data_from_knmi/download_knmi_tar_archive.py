import logging
import sys
from datetime import datetime
from pathlib import Path
import os

import requests

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

api_url = "https://api.dataplatform.knmi.nl/open-data"
api_version = "v1"

# Parameters
api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjQzYTE3NTczZjYwMTRjMDM5YTNkNzE4Mjk2NzJiMDcyIiwiaCI6Im11cm11cjEyOCJ9"
dataset_name = "radar_tar_refl_composites"
dataset_version = "1.0"
max_keys = "200"

savefolder = r"/DATAFOLDER/cluster_projects/pr/0949_10/reflectivity_tar_archive_20210408"

# Use list files request to request first 10 files of the day.
# timestamp = datetime.utcnow().date().strftime("%Y%m%d")
# start_after_filename_prefix = f"KMDS__OPER_P___10M_OBS_L2_{timestamp}"
start_after_filename_prefix = f"RAD25_OPER_R___TARPCP__L2__20201225T000000_20211226T000000_0001"
list_files_response = requests.get(
    f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files",
    headers={"Authorization": api_key},
    params={"maxKeys": max_keys, "startAfterFilename": start_after_filename_prefix},
)
list_files = list_files_response.json()

logger.info(f"List files response:\n{list_files}")
dataset_files = list_files.get("files")

# Retrieve first file in the list files response
for i in range(int(max_keys)):
    filename = dataset_files[i].get("filename")
    print (filename)
    logger.info(f"Retrieve file with name: {filename}")
    endpoint = f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    get_file_response = requests.get(endpoint, headers={"Authorization": api_key})
    if get_file_response.status_code != 200:
        logger.error("Unable to retrieve download url for file")
        logger.error(get_file_response.text)
        sys.exit(1)
    
    download_url = get_file_response.json().get("temporaryDownloadUrl")
    dataset_file_response = requests.get(download_url)
    if dataset_file_response.status_code != 200:
        logger.error("Unable to download file using download URL")
        logger.error(dataset_file_response.text)
        sys.exit(1)
    
    # Write dataset file to disk
    p = Path(os.path.join(savefolder, filename))
    p.write_bytes(dataset_file_response.content)
    logger.info(f"Successfully downloaded dataset file to {p}")
