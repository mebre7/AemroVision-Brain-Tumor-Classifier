from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.utils.common import logger

STAGE_NAME = "Data Ingestion Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>> Stage \'{STAGE_NAME}\' is started. <<<<<<<<<<<<<<<")
        stage1 = DataIngestionPipeline()
        stage1.main()
        logger.info(f">>>>>>>>>>>>>>> Stage \'{STAGE_NAME}\'is completed. <<<<<<<<<<<<<<<\n\n x=================x\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Prepare Base Model Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>> Stage \'{STAGE_NAME}\' is started. <<<<<<<<<<<<<<<")
        stage1 = PrepareBaseModelPipeline()
        stage1.main()
        logger.info(f">>>>>>>>>>>>>>> Stage \'{STAGE_NAME}\'is completed. <<<<<<<<<<<<<<<\n\n x=================x\n\n")
    except Exception as e:
        logger.exception(e)
        raise e