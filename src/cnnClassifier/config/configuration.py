import os
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)
from cnnClassifier.utils.common import (
    read_yaml, create_directories
)
from cnnClassifier.constants import *

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)], verbose=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )
    
    def get_training_config(self) -> TrainingConfig:
        params = self.params
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        raw_data = self.config.data_ingestion.unzip_dir
        create_directories([training.root_dir])
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(os.path.join(raw_data, "Brain_Tumor_MRI_dataset", "Training")),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        params = self.params
        eval_config = self.config.evaluation
        training_config = self.config.training
        raw_data = self.config.data_ingestion.unzip_dir
        create_directories([eval_config.root_dir])
        return EvaluationConfig(
            root_dir=Path(eval_config.root_dir),
            path_of_model=Path(training_config.trained_model_path),
            training_data=Path(os.path.join(raw_data, "Brain_Tumor_MRI_dataset", "Testing")),
            all_params=params,
            metrics_file_path=Path(eval_config.metrics_file_path),
            mlflow_uri=eval_config.mlflow_uri,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE
        )