import os
from .CustomDataPreparation import LeukemiaDataPreparation
from .CustomDataPreparation import LeukemiaALLAMLDataPreparation
import pandas as pd

class DataPreparationFactory:
    def __init__(self, dataset_name, input_folder, output_folder, cluster_strategy, cluster_parameters=None, cluster_on_correlation=False, label_column=None):
        """
        Initialize the DataPreparationFactory.

        :param dataset_name: Name of the dataset.
        :param input_folder: Path to the input folder containing the dataset.
        :param output_folder: Path to the output folder where results will be saved.
        :param cluster_strategy: Strategy to use for clustering features (e.g., 'hierarchical', 'correlation').
        :param label_column: Name of the column containing labels.
        """
        self.dataset_name = dataset_name
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cluster_strategy = cluster_strategy
        self.cluster_parameters = cluster_parameters
        self.cluster_on_correlation = cluster_on_correlation
        self.label_column = label_column
        self.label_balance_string = 'Label balance: '
        self.export = True
        
    def _check_output_folder(self):
        """
        Controlla se la cartella di output esiste e contiene già i dati del dataset.
        Ignora file di log come log_iteration.txt.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Lista di file da ignorare
        files_to_ignore = ['log_iteration.txt', 'loggers.log', 'shap_scores']
        
        # Verifica se ci sono file rilevanti nella cartella (escludendo quelli da ignorare)
        relevant_files = [f for f in os.listdir(self.output_folder) 
                        if not any(ignore in f for ignore in files_to_ignore)]
        
        
        if relevant_files:
            # Verifica se esiste il file di informazioni del dataset
            info_file_path = os.path.join(self.output_folder, 'dataset_info.txt')
            
            if not os.path.exists(info_file_path):
                raise ValueError(f"La cartella di output '{self.output_folder}' contiene file ma non è stato trovato dataset_info.txt.")
            else:
                # Verifica se le informazioni nel file corrispondono alla configurazione corrente
                expected_info = (f"Dataset Name: {self.dataset_name}\n"
                                f"Cluster Strategy: {self.cluster_strategy}\n"
                                f"Label Column: {self.label_column}\n")
                
                with open(info_file_path, 'r') as file:
                    info_content = file.read()
                    
                # Controlla se le prime tre righe corrispondono (ignora il bilanciamento delle label)
                info_lines = info_content.split('\n')[:3]
                expected_lines = expected_info.split('\n')[:3]
                
                if info_lines != expected_lines:
                    # Le configurazioni non corrispondono, elimina i file rilevanti
                    print("La configurazione del dataset è cambiata, i file precedenti verranno rimossi.")
                    for filename in relevant_files:
                        file_path = os.path.join(self.output_folder, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                else:
                    # La configurazione corrisponde, utilizza i file esistenti
                    self.export = False
                    print(f"Trovato il file dataset_info.txt che corrisponde alla configurazione corrente:\n{info_content}")
                    
                    # Verifica se esistono i file di dati necessari
                    data_file = os.path.join(self.output_folder, "data.csv")
                    clusters_file = os.path.join(self.output_folder, "clusters.csv")
                    
                    if not (os.path.exists(data_file) and os.path.exists(clusters_file)):
                        print("Mancano i file data.csv o clusters.csv. I dati verranno riesportati.")
                        self.export = True
                        
    def _save_dataset_info(self): 
        info_file_path = os.path.join(self.output_folder, 'dataset_info.txt')
        with open(info_file_path, 'w') as file:
            file.write(f"Dataset Name: {self.dataset_name}\n")
            file.write(f"Cluster Strategy: {self.cluster_strategy}\n")
            file.write(f"Cluster on Correlation: {self.cluster_on_correlation}\n")
            file.write(f"Label Column: {self.label_column}\n")
            file.write(self.label_balance_string + "\n")
            
    def _prepare_data__training(self, data, clusters):
        
        labels = data[self.label_column].copy()
    
        if not set(labels.unique()).issubset({0, 1}):
            raise ValueError("I label devono essere 0 o 1.")
        
        self.label_balance_string = f"Label counts - class 0: {(labels == 0).sum()}, class 1: {(labels == 1).sum()}"
        
        features_data = data.drop(['cod_pz', self.label_column], axis=1).copy()
        features = features_data.columns.values
        
        return data, features_data, features, labels

    def export_data_clusters(self):
        self._check_output_folder()
        
        if self.export == False:
            
            data_path = os.path.join(self.output_folder, "data.csv")
            clusters_path = os.path.join(self.output_folder, "clusters.csv")
            
            data = pd.read_csv(data_path)
            clusters = pd.read_csv(clusters_path)
            data, features_data, features, labels = self._prepare_data__training(data, clusters)

        else:

            if self.dataset_name == "Leukemia":
                data_preparation = LeukemiaDataPreparation(self.input_folder, self.output_folder, self.cluster_strategy, self.cluster_parameters, self.cluster_on_correlation, label_column=self.label_column)
            elif self.dataset_name == "leukemia_ALL_AML":
                if self.cluster_strategy is None:
                    raise ValueError("cluster_strategy must be provided for leukemia_ALL_AML dataset")
                data_preparation = LeukemiaALLAMLDataPreparation(self.input_folder, self.output_folder, self.cluster_strategy, self.cluster_parameters, self.cluster_on_correlation, label_column=self.label_column)
            else:
                raise ValueError(f"Unknown dataset name: {self.dataset_name}")
            
            data, clusters = data_preparation.export_data_clusters()
            
            data, features_data, features, labels = self._prepare_data__training(data, clusters)
            
            self._save_dataset_info()
                
        return {
            "data": data,
            "features_data": features_data,
            "features": features,
            "labels": labels,
            "clusters": clusters,
            "clusters_over_features": clusters.copy(),
        }
            
if __name__ == "__main__":
    
    # Carlo_dataset
    input_folder = "/home/edofroses/genomics/Raw_Datasets/Carlo_data"
    output_folder = "/home/edofroses/genomics/dataset_training_kmeans"
    cluster_strategy = "KMeans"
    cluster_parameters = 'searching'
    #cluster_parameters = {'distance_metric': 'mahalanobis', 'linkage_method': 'single'}
    label_column = "TTFT"
    
    data_preparation_factory = DataPreparationFactory("Leukemia", input_folder, output_folder, cluster_strategy, cluster_parameters, label_column)
    data_preparation_factory.export_data_clusters()
    
    """
    # leukemia_ALL_AML
    input_folder = "data/leukemia_ALL_AML"
    output_folder = "data/leukemia_ALL_AML_output"
    cluster_strategy = "hierarchical"
    label_column = "cancer"
    
    data_preparation_factory = DataPreparationFactory("leukemia_ALL_AML", input_folder, output_folder, cluster_strategy, label_column)
    data_preparation_factory.export_data_clusters()
    
    # leukemia_ALL_AML
    input_folder = "data/leukemia_ALL_AML"
    output_folder = "data/leukemia_ALL_AML_output"
    cluster_strategy = "correlation"
    label_column = "cancer"
    
    data_preparation_factory = DataPreparationFactory("leukemia_ALL_AML", input_folder, output_folder, cluster_strategy, label_column)
    data_preparation_factory.export_data_clusters()
    """