from cnnImageClassification.constants import *
from cnnImageClassification.utils.common import read_yaml, create_directories
from cnnImageClassification.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig

class ConfigManager:
    def __init__(
            self, 
            config__filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config__filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root]) 
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
    
    def get_prep_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.CLASSES,
            params_loss=self.params.LOSS,
            params_metrics=self.params.METRICS,
            params_epochs=self.params.EPOCHS  # Ensure this line is added
        )
