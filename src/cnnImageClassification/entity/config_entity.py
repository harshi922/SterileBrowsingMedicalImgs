from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_image_size: list
    params_batch_size: int
    params_batch_size: int
    params_classes: int
    params_loss: str
    params_metrics: list
    params_epochs: int
