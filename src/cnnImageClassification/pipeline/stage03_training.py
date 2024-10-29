from cnnImageClassification.config.configuration import ConfigManager
from cnnImageClassification.components.training import Training
from cnnImageClassification.components.prepare_callbacks import PrepareCallback 
from cnnImageClassification import logger


STAGE_NAME = 'Training Stage'

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.prepare_training_testing_data()
        training.train(
            callback_list=callback_list
        )
if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.exception(e)
        raise e