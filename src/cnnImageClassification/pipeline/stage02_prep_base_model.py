from cnnImageClassification.config.configuration import ConfigManager
from cnnImageClassification.components.prep_base_model import PrepareBaseModel
from cnnImageClassification import logger

STAGE_NAME = "Prepare Model Stage"

class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        prepare_base_model_config = config.get_prep_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.define_base_model()
        prepare_base_model.create_base_model()

if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = PrepareModelTrainingPipeline()
        obj.main()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.exception(e)
        raise e