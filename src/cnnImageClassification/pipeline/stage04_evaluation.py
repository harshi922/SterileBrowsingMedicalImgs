from cnnImageClassification.config.configuration import ConfigManager
from cnnImageClassification.components.evaluation import Evaluation
from cnnImageClassification import logger   

STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()

if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.exception(e)
        raise e