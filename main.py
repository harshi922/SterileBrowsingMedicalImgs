from cnnImageClassification import logger
from cnnImageClassification.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from cnnImageClassification.pipeline.stage02_prep_base_model import PrepareModelTrainingPipeline
from cnnImageClassification.pipeline.stage03_training import TrainingPipeline
from cnnImageClassification.pipeline.stage04_evaluation import EvaluationPipeline

try:
    STAGE_NAME = "Data Ingestion Stage"
    logger.info(f"Starting {STAGE_NAME}")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"Completed {STAGE_NAME}")
    STAGE_NAME = "Prepare Model Stage"
    logger.info(f"Starting {STAGE_NAME}")
    obj = PrepareModelTrainingPipeline()
    obj.main()
    logger.info(f"Completed {STAGE_NAME}")
    STAGE_NAME = "Training Stage"
    logger.info(f"Starting {STAGE_NAME}")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f"Completed {STAGE_NAME}")
    STAGE_NAME = "Evaluation Stage"
    logger.info(f"Starting {STAGE_NAME}")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f"Completed {STAGE_NAME}")
except Exception as e:
    logger.exception(e)
    raise e
