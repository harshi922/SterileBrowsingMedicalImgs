import os
import urllib.request as request
import zipfile
from cnnImageClassification import logger
from cnnImageClassification.utils.common import get_size
from pathlib import Path

from cnnImageClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(url=self.config.source_URL, filename=self.config.local_data_file)
            logger.info(f"Downloading file{file_name} with info {headers}")
        else:
            logger.info(f"File already exists with size {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Download the zip file from the source URL
        return None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)        

