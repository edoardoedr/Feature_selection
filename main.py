from FeSCAE.DataPreparation import DataPreparationFactory
from FeSCAE.tools import Config, LoggerFeSCAE, get_scaler
from FeSCAE.Clustering import GetClustering
from FeSCAE.Filters import create_filter
import os
import pandas as pd





if __name__ == "__main__":
    
    config_path = os.path.join("config_yaml", "base_config.yaml")
    
    config = Config(config_path)
    logger = LoggerFeSCAE(config.output.output_folder)
    logger.log_config(config)
    scaler = get_scaler(config.dataset.scaler)

    logger.log_section("CARICAMENTO DATI")
    data_preparation = DataPreparationFactory(dataset_name=config.dataset.name,
                                              input_folder=config.dataset.input_folder,
                                              output_folder=config.output.output_folder,
                                              label_column=config.dataset.label_column)

    data = data_preparation.export_data()

    logger.log_section("CLUSTERING...")

    clustering = GetClustering(data = data['features_data'], 
                               output_folder=config.output.output_folder, 
                               cluster_strategy=config.clustering.strategy, 
                               cluster_parameters=config.clustering.parameters, 
                               cluster_on_correlation=config.clustering.cluster_on_correlation,
                               number_of_clusters=config.clustering.number_of_clusters)

    clusters = clustering.get_clusters()


    features = data['features'].copy()
    features_data = data['features_data'][features].copy()
    iteration = 0

    logger.log_section("FILTERING...")

    FeatureFilter = create_filter(config.filters, iteration, config.output.output_folder, scaler)
    
    print(f'feature data shape before filtering: {features_data.shape}')
    print(f'clusters shape: {clusters.shape}')
            
    selected_features = FeatureFilter.filter(features_data, clusters) 




