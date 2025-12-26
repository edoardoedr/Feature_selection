from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
from ..ClusteringTools import (group_features_by_hierarchical_clustering, params_search_hierarchical_clustering,
                             group_features_by_kmeans, params_search_kmeans,
                             group_features_by_dbscan, params_search_dbscan,
                             group_features_by_kmedoids, params_search_kmedoids)

class BaseDataPreparation(ABC):
    def __init__(self, input_folder, output_folder, cluster_strategy, cluster_parameters, cluster_on_correlation, label_column):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cluster_strategy = cluster_strategy
        self.cluster_parameters = cluster_parameters
        self.cluster_on_correlation = cluster_on_correlation
        self.label_column = label_column

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_data_clusters(self):
        pass

    def export_data_clusters(self):
        
        data, clusters = self.process_data_clusters()

        data.to_csv(os.path.join(self.output_folder, "data.csv"), index=False)
        clusters.to_csv(os.path.join(self.output_folder, "clusters.csv"), index=False)
        
        return data, clusters

    def get_clusters(self, df):
        
        #qui aggiungiamo il calcolo della correlazione se cluster on correlation è True
        if self.cluster_on_correlation:
            numeric_df = df.select_dtypes(include='number')
            # Calcola la matrice di correlazione con numpy
            corr_matrix = np.corrcoef(numeric_df.T)
            # Trasforma la matrice in un DataFrame mantenendo indici numerici e nomi delle colonne corretti
            correlation_df = pd.DataFrame(data=corr_matrix, columns=numeric_df.columns)
            
            df = correlation_df

        if self.cluster_strategy == 'hierarchical':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_hierarchical_clustering(df, output_dir=self.output_folder)
                return group_features_by_hierarchical_clustering(df, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder)
            else:
                return group_features_by_hierarchical_clustering(df, self.cluster_parameters, mode='clustering', output_dir=self.output_folder)

        elif self.cluster_strategy == 'DBScan':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_dbscan(df, output_dir=self.output_folder)
                return group_features_by_dbscan(df, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder)
            else:
                return group_features_by_dbscan(df, self.cluster_parameters, mode='clustering', output_dir=self.output_folder)

        elif self.cluster_strategy == 'KMeans':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_kmeans(df, output_dir=self.output_folder)
                return group_features_by_kmeans(df, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder)
            else:
                return group_features_by_kmeans(df, self.cluster_parameters, mode='clustering', output_dir=self.output_folder)

        elif self.cluster_strategy == 'KMedoids':
            if self.cluster_parameters == "searching":
                self.cluster_parameters = params_search_kmedoids(df, output_dir=self.output_folder)
                return group_features_by_kmedoids(df, self.cluster_parameters['params'], mode='clustering', output_dir=self.output_folder)
            else:
                return group_features_by_kmedoids(df, self.cluster_parameters, mode='clustering', output_dir=self.output_folder)

        else:
            raise ValueError(f"Unknown cluster strategy: {self.cluster_strategy}")