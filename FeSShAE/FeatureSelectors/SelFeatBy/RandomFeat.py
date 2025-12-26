import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class RandomFeat:
    """
    Selettore di feature che assegna valori random alle feature e seleziona il primo quartile.
    Questa classe è utile come baseline per confrontare l'efficacia di metodi più sofisticati.
    """
    
    def __init__(self, output_dir, logger=None,):
        """
        Inizializza il selettore di feature random.
        
        Args:
            output_dir (str): Directory dove salvare i risultati
            logger: Logger per registrare informazioni (opzionale)
            shap_sampling: Parametro mantenuto per compatibilità con ShapSelector
        """
        self.output_dir = output_dir
        self.logger = logger
        # Il parametro shap_sampling è ignorato ma mantenuto per compatibilità
    
    def select_features(self, model, data, scaler=None, iteration=0):
        """
        Assegna valori casuali alle feature e seleziona quelle nel primo quartile.
        
        Args:
            model: Modello addestrato (non utilizzato, mantenuto per compatibilità)
            data (DataFrame): DataFrame contenente le feature
            scaler: Scaler per normalizzare i dati (non utilizzato, mantenuto per compatibilità)
            iteration (int): Numero dell'iterazione corrente
            
        Returns:
            tuple: (lista delle feature selezionate, DataFrame con i valori random)
        """
        if self.logger:
            self.logger.log_message(f"Avvio selezione random delle feature (Iterazione {iteration})")
            self.logger.log_message(f"Numero di feature totali: {data.shape[1]}")
        
        # Crea directory per i risultati
        results_dir = os.path.join(self.output_dir, f'random_selection_iteration_{iteration}')
        os.makedirs(results_dir, exist_ok=True)
        
        features = data.columns.tolist()
        
        # Genera valori random per ogni feature
        random_values = np.random.uniform(0, 1, size=len(features))
        
        # Crea DataFrame con i valori random
        random_scores = pd.DataFrame({
            'Feature': features,
            'Random_Value': random_values
        })
        
        # Ordina per valore random (decrescente)
        random_scores = random_scores.sort_values('Random_Value', ascending=False)
        
        # Calcola il primo quartile (25%) delle feature
        num_to_select = max(1, int(len(features) * 0.25))
        quartile_value = np.percentile(random_values, 75)  # valore del 75° percentile
        
        # Seleziona le feature nel primo quartile
        selected_features = random_scores.head(num_to_select)['Feature'].tolist()
        
        if self.logger:
            self.logger.log_message(f"Selezionate {len(selected_features)} feature (primo quartile)")
            self.logger.log_message(f"Valore di soglia (75° percentile): {quartile_value:.4f}")
        
        # Salva i risultati
        self._save_results(random_scores, selected_features, results_dir, iteration)
        
        # Visualizza la distribuzione
        self._plot_distribution(random_scores, quartile_value, results_dir, iteration)
        
        return selected_features, random_scores
    
    def _save_results(self, random_scores, selected_features, results_dir, iteration):
        """
        Salva i risultati della selezione.
        
        Args:
            random_scores (DataFrame): DataFrame con i valori random delle feature
            selected_features (list): Lista delle feature selezionate
            results_dir (str): Directory dove salvare i risultati
            iteration (int): Numero dell'iterazione
        """
        # Marca le feature selezionate
        random_scores['Selected'] = random_scores['Feature'].isin(selected_features)
        
        # Salva il DataFrame completo
        random_scores.to_excel(
            os.path.join(results_dir, f'random_selection_scores_{iteration}.xlsx'),
            index=False
        )
        
        # Salva solo le feature selezionate
        pd.DataFrame({'Feature': selected_features}).to_excel(
            os.path.join(results_dir, f'random_selected_features_{iteration}.xlsx'),
            index=False
        )
        
        if self.logger:
            self.logger.log_message(f"Risultati salvati in: {results_dir}")
    
    def _plot_distribution(self, random_scores, threshold, results_dir, iteration):
        """
        Crea e salva un grafico della distribuzione dei valori random.
        
        Args:
            random_scores (DataFrame): DataFrame con i valori random delle feature
            threshold (float): Valore soglia per la selezione
            results_dir (str): Directory dove salvare il grafico
            iteration (int): Numero dell'iterazione
        """
        plt.figure(figsize=(12, 8))
        
        # Crea l'istogramma dei valori random
        plt.hist(random_scores['Random_Value'], bins=20, alpha=0.7, color='skyblue')
        
        # Aggiunge una linea verticale per indicare la soglia
        plt.axvline(x=threshold, color='red', linestyle='--', 
                    label=f'Soglia (75° percentile): {threshold:.4f}')
        
        # Calcola statistiche della distribuzione
        mean_val = random_scores['Random_Value'].mean()
        median_val = random_scores['Random_Value'].median()
        
        # Aggiunge linee per media e mediana
        plt.axvline(x=mean_val, color='green', linestyle='-', 
                    label=f'Media: {mean_val:.4f}')
        plt.axvline(x=median_val, color='purple', linestyle=':', 
                    label=f'Mediana: {median_val:.4f}')
        
        plt.title('Distribuzione dei valori random per le feature')
        plt.xlabel('Valore random')
        plt.ylabel('Frequenza')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Aggiunge un riquadro con le informazioni sulla selezione
        selected_count = random_scores['Selected'].sum()
        total_count = len(random_scores)
        info_text = (f"Feature selezionate: {selected_count}/{total_count}\n"
                    f"Percentuale: {selected_count/total_count*100:.1f}%")
        
        plt.annotate(info_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.5", 
                                          fc="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'random_distribution_{iteration}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.log_message(f"Grafico di distribuzione salvato in: {results_dir}")
        
    def get_name(self):
        """Ritorna il nome del selettore."""
        return "RandomFeat"