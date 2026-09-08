from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = ModelTraining(training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == "__main__":
    try:
        logger.info("Starting Model Training Pipeline")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
    except Exception as e:
        raise e