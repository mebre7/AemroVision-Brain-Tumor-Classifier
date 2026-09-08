from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion = DataIngestion(config.get_data_ingestion_config())
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info("Starting Data Ingestion Pipeline")
        pipeline = DataIngestionPipeline()
        pipeline.main()
    except Exception as e:
        raise e