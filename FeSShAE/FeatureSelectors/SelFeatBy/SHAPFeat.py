import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

class SHAPFeat:
    """
    Classe per la selezione di feature tramite analisi SHAP.
    Supporta tutti i tipi di classificatori (NN, XGBoost, RandomForest, SVM).
    """
    
    def __init__(self, output_dir, logger, shap_sampling=100):
        """
        Inizializza il selettore SHAP.
        
        Args:
            output_folder (str): Cartella di output per i risultati
            shap_sampling (int): Numero di campioni da utilizzare per l'analisi SHAP
        """
        self.output_dir = output_dir
        self.shap_sampling = shap_sampling
        self.logger = logger
        
        # Crea la directory per i risultati SHAP se non esiste
        self.shap_scores_dir = ''
    
    def select_features(self, model, data, scaler, iteration):
        """
        Seleziona le feature più importanti utilizzando SHAP.
        
        Args:
            model: Modello di classificazione (deve avere un metodo get_shap_values)
            data (DataFrame): DataFrame con i dati di input
            scaler: Scaler utilizzato per normalizzare i dati
            iteration: Numero dell'iterazione corrente
            mode (str): Modalità ('DEBUG' o 'PROD')
            
        Returns:
            tuple: (feature selezionate, DataFrame con punteggi SHAP)
        """
        self.shap_scores_dir = os.path.join(self.output_dir, f'output_iteration_{iteration}')
        os.makedirs(self.shap_scores_dir, exist_ok=True)
        # Normalizzazione dei dati
        data_normalized = pd.DataFrame(scaler.transform(data), columns=data.columns)
        
        # Campionamento per SHAP
        data_sampled = data_normalized.sample(min(self.shap_sampling, len(data_normalized)))
        
        # Ottieni i valori SHAP dal modello
        shap_values = model.get_shap_values(data_sampled)
        
        # Calcola l'intensità SHAP (importanza della feature) usando il valore assoluto
        feature_importance_0 = np.abs(shap_values[0]).mean(axis=0)
        feature_importance_1 = np.abs(shap_values[1]).mean(axis=0)
        feature_importance = feature_importance_0 + feature_importance_1
        # Crea un DataFrame con le feature e la loro importanza
        feature_importance_df = pd.DataFrame({'Feature': data_sampled.columns, 'Intensity': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Intensity', ascending=False)
        
        # Calcola il threshold per il primo quartile
        intensity_threshold = feature_importance_df['Intensity'].quantile(0.25)
        
        # Filtra le feature in base al threshold
        selected_features_df = feature_importance_df[feature_importance_df['Intensity'] >= intensity_threshold]
        eliminated_features_df = feature_importance_df[feature_importance_df['Intensity'] < intensity_threshold]
        
        # Ottieni la lista delle feature selezionate
        selection = selected_features_df['Feature'].values
        
        # Salva i risultati in un file Excel
        with pd.ExcelWriter(os.path.join(self.shap_scores_dir, f'shap_scores_{iteration}.xlsx')) as writer:
            selected_features_df.to_excel(writer, sheet_name='Winners', index=False)
            eliminated_features_df.to_excel(writer, sheet_name='Eliminated', index=False)
        
        # Crea grafici SHAP (opzionale)
        self._plot_shap_summary(shap_values[1], data_sampled, iteration)
        
        return selection, selected_features_df
    
    def _plot_shap_summary(self, shap_values, data, iteration):
        """
        Crea e salva il grafico di riepilogo SHAP.
        
        Args:
            shap_values (DataFrame): DataFrame con i valori SHAP
            data (DataFrame): DataFrame con i dati di input
            iteration: Numero dell'iterazione
        """
        plt.figure(figsize=(12, 8))
        
        # Crea un oggetto shap.Explanation
        shap_array = shap_values.to_numpy()
        feature_names = data.columns.tolist()
        
        explanation = shap.Explanation(
            values=shap_array,
            data=data.to_numpy(),
            feature_names=feature_names
        )
        
        # Plot summary
        shap.summary_plot(explanation, data, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.shap_scores_dir, f'shap_summary_{iteration}.png'), dpi=500, bbox_inches='tight')
        plt.close()