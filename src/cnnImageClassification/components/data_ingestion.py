import os
import zipfile
from cnnImageClassification import logger
from cnnImageClassification.utils.common import get_size
from pathlib import Path
import gdown

from cnnImageClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            gdown.download(f'https://drive.google.com/uc?id={self.config.source_URL.split("/")[-2]}', self.config.local_data_file, quiet=False)
            logger.info(f"Downloading file{self.config.local_data_file}")
        else:
            print("self.config.source_URL", self.config.source_URL.split('view')[-1])
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

