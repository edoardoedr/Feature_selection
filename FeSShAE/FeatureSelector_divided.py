import os
import argparse
import pandas as pd
from .Classifiers import create_classifier
from .tools import Config, LoggerFeSShAE
from .DataPreparation import DataPreparationFactory

def main(config_path, features_path=None):
    """
    Esegue il training di un classificatore su feature già selezionate.
    
    Args:
        config_path (str): Percorso del file di configurazione
        features_path (str): Percorso del file con le feature selezionate (opzionale)
    """
    # Carica configurazione
    config = Config(config_path)
    logger = LoggerFeSShAE(config.output.output_folder)
    logger.log_message("Inizializzazione classificatore con features preselezionate")
    
    # Carica le feature selezionate
    if features_path is None:
        features_path = os.path.join(config.output.output_folder, 'selected_features.xlsx')
    
    if not os.path.exists(features_path):
        logger.log_message(f"File delle feature non trovato: {features_path}")
        return
    
    selected_features = pd.read_excel(features_path)['features'].tolist()
    logger.log_message(f"Caricate {len(selected_features)} caratteristiche da {features_path}")
    
    # Carica i dati
    logger.log_section("CARICAMENTO DATI")
    data_preparation = DataPreparationFactory(
        dataset_name=config.dataset.name,
        input_folder=config.dataset.input_folder,
        output_folder=config.output.output_folder,
        label_column=config.dataset.label_column
    )
    data = data_preparation.export_data()
    
    # Filtra i dati usando solo le feature selezionate
    if all(feature in data['data'].columns for feature in selected_features):
        final_data = data['data'][selected_features]
        labels = data['labels']
        logger.log_message(f"Dati filtrati con {len(selected_features)} caratteristiche")
    else:
        missing_features = [f for f in selected_features if f not in data['data'].columns]
        logger.log_message(f"ERRORE: Alcune caratteristiche non sono presenti nel dataset: {missing_features[:5]}...")
        return
    
    # Training e valutazione del classificatore
    logger.log_section("VALUTAZIONE DEL MODELLO")
    
    classifier_type = config.classifier.classifier_type
    logger.log_message(f"Valutazione modello {classifier_type} su {len(selected_features)} caratteristiche")
    
    classifier = create_classifier(
        classifier_type,
        config.classifier.param_space[classifier_type],
        config.classifier.cv,
        config.output.output_folder,
        logger=logger
    )
    
    results = classifier.evaluate(final_data, labels)
    
    # Stampa risultati finali
    logger.log_section("RISULTATI FINALI")
    for metric, value in results.items():
        if isinstance(value, (float, int)):
            logger.log_message(f"{metric}: {value:.4f}")
    
    logger.log_completion()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training con feature preselezionate")
    parser.add_argument("config", help="Path to the configuration YAML file")
    parser.add_argument("--features", help="Path to the selected features file (Excel format)")
    args = parser.parse_args()
    
    main(args.config, args.features)