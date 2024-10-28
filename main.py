from cnnImageClassification import logger
from cnnImageClassification.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"Starting {STAGE_NAME}")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"Completed {STAGE_NAME}")
except Exception as e:
    logger.exception(e)
    raise e
