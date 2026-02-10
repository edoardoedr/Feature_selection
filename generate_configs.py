"""
Script per generare file di configurazione multipli per esperimenti.
Varia dataset, numero di cluster e strategia di clustering.
"""

import os
import yaml

# Definizione dei parametri da variare
DATASETS = {
    "Colon": "label_colon",
    "Prostate-GE": "label_prostate",
    "Lymphoma": "label_lymphoma",
    "Leukemia": "label_leukemia",
    "TOX-171": "label_tox171",
    "GLI-85": "label_gli85"
}

CLUSTERS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
STRATEGIES = ["Hierarchical", "KMeans"]

# Template di configurazione base
BASE_CONFIG = {
    "dataset": {
        "name": None,
        "input_folder": "Raw_Datasets/",
        "label_column": None,
        "scaler": "standard"
    },
    "output": {
        "output_folder": None,
        "logs_folder": None
    },
    "clustering": {
        "strategy": None,
        "cluster_on_correlation": True,
        "number_of_clusters": None,
        "range_n_clusters": 0,
        "parameters": "searching"
    },
    "filters": {
        "filter_type": "AE",
        "autoencoder_parameters": {
            "lr": 0.001,
            "epochs": 100,
            "AE_name": "LinearAE",
            "n_layers": [1, 2, 3, 4]
        },
        "medoid_parameters": {
            "distance_metric": "euclidean"
        }
    }
}


def generate_configs():
    """Genera tutti i file di configurazione."""
    
    configs_dir = "configs_paper"
    os.makedirs(configs_dir, exist_ok=True)
    
    total_configs = len(DATASETS) * len(CLUSTERS) * len(STRATEGIES)
    print(f"Generazione di {total_configs} file di configurazione...")
    
    count = 0
    for dataset_name, label_column in DATASETS.items():
        # Normalizza il nome del dataset per i path (rimuovi caratteri speciali)
        dataset_clean = dataset_name.lower().replace("-", "")
        
        for n_clusters in CLUSTERS:
            for strategy in STRATEGIES:
                # Crea una copia del template
                config = yaml.safe_load(yaml.dump(BASE_CONFIG))
                
                # Nome strategia in minuscolo per i path
                strategy_lower = strategy.lower()
                
                # Imposta i valori specifici
                config["dataset"]["name"] = dataset_name
                config["dataset"]["label_column"] = label_column
                
                output_path = f"output_experiments/{dataset_clean}/output_{dataset_clean}_{n_clusters}_{strategy_lower}"
                config["output"]["output_folder"] = output_path
                config["output"]["logs_folder"] = output_path
                
                config["clustering"]["strategy"] = strategy
                config["clustering"]["number_of_clusters"] = n_clusters
                
                # Nome del file di configurazione
                config_filename = f"config_{dataset_clean}_{n_clusters}_{strategy_lower}.yaml"
                os.makedirs(os.path.join(configs_dir, dataset_clean), exist_ok=True)
                config_path = os.path.join(configs_dir, dataset_clean, config_filename)
                
                # Salva il file di configurazione
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                count += 1
                if count % 20 == 0:
                    print(f"  Generati {count}/{total_configs} file...")
    
    print(f"\n✓ Generati con successo {count} file di configurazione in '{configs_dir}/'")
    
    # Stampa un riepilogo
    print("\nRiepilogo:")
    print(f"  - Dataset: {len(DATASETS)} ({', '.join(DATASETS.keys())})")
    print(f"  - Cluster: {len(CLUSTERS)} ({min(CLUSTERS)}-{max(CLUSTERS)})")
    print(f"  - Strategie: {len(STRATEGIES)} ({', '.join(STRATEGIES)})")
    print(f"  - Totale: {count} configurazioni")
    
    # Crea le cartelle di output
    print("\nCreazione struttura cartelle output...")
    for dataset_name in DATASETS.keys():
        dataset_clean = dataset_name.lower().replace("-", "")
        dataset_dir = f"output_experiments/{dataset_clean}"
        os.makedirs(dataset_dir, exist_ok=True)
    
    print("✓ Struttura cartelle creata")
    

if __name__ == "__main__":
    generate_configs()
