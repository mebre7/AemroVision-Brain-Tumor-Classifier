from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion = DataIngestion(config.get_data_ingestion_config())
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()