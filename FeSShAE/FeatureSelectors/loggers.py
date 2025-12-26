import os
import time
from datetime import datetime
import numpy as np

class SearchSelectorLogger:
    """
    Logger semplificato che registra i risultati per ogni combinazione di iperparametri
    e fold di cross-validation, e alla fine mostra un riepilogo del miglior modello.
    """
    
    def __init__(self, iteration, output_dir='best_models'):
        """
        Inizializza il logger.
        
        Args:
            iteration (int/str): Iterazione corrente o identificatore (es. 'WINNER')
            output_dir (str): Directory base dove salvare i log
        """
        self.model_dir = os.path.join(output_dir, f'output_iteration_{iteration}')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.model_dir, 'classifier_training_log.txt')
        self.start_time = time.time()
        self.separator = "=" * 50
        
        # Intestazione del file di log
        with open(self.log_path, 'w') as f:
            f.write(f"=== TRAINING LOG - ITERAZIONE {iteration} ===\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_message(self, message):
        """Registra un messaggio generico nel log"""
        with open(self.log_path, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    def log_model_evaluation(self, model_idx, total_models, hyperparams, model_type, scores, mean_acc):
        """
        Registra i risultati della valutazione di un modello.
        
        Args:
            model_idx (int): Indice del modello
            total_models (int): Numero totale di modelli
            hyperparams (tuple): Gli iperparametri del modello
            model_type (str): Tipo di modello (es. 'neural_network')
            scores (list): Punteggi delle fold di cross-validation
            mean_acc (float): Accuratezza media
        """
        message = (
            f"Modello {model_idx}/{total_models} - Tipo: {model_type}\n"
            f"Iperparametri: {hyperparams}\n"
            f"Accuratezza per fold: {[round(s, 4) for s in scores]}\n"
            f"Accuratezza media: {mean_acc:.4f}\n"
        )
        self.log_message(message)
    
    def log_cv_fold(self, cv_idx, total_cv, acc):
        """
        Registra il risultato di una singola fold di cross-validation.
        
        Args:
            cv_idx (int): Indice della fold
            total_cv (int): Numero totale di fold
            acc (float): Accuratezza della fold
        """
        self.log_message(f"  Fold {cv_idx}/{total_cv} - Acc: {acc:.4f}")
    
    def log_best_model(self, best_model_idx, best_hyperparams, best_acc, best_model_type, best_scores, ci_lower, ci_upper):
        """
        Registra i dettagli del miglior modello trovato.
        
        Args:
            best_model_idx (int): Indice del miglior modello
            best_hyperparams (tuple): Gli iperparametri del miglior modello
            best_acc (float): Accuratezza del miglior modello
            best_model_type (str): Tipo del miglior modello
            best_scores (list): Punteggi delle fold di cross-validation del miglior modello
            ci_lower (float): Limite inferiore dell'intervallo di confidenza
            ci_upper (float): Limite superiore dell'intervallo di confidenza
        """
        message = (
            f"\n{self.separator}\n"
            "MIGLIOR MODELLO\n" + 
            f"{self.separator}\n"
            f"Modello #{best_model_idx} - Tipo: {best_model_type}\n"
            f"Iperparametri: {best_hyperparams}\n"
            f"Accuratezza media: {best_acc:.4f}\n"
            f"Intervallo di confidenza 95%: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
            f"Accuratezza per fold: {[round(s, 4) for s in best_scores]}\n"
            f"{self.separator}\n"
        )
        self.log_message(message)
    
    def log_shap_selection(self, selected_features):
        """
        Registra le feature selezionate mediante SHAP.
        
        Args:
            selected_features (list): Lista delle feature selezionate
        """
        message = (
            f"\n{self.separator}\n"
            "SELEZIONE FEATURE CON SHAP\n" + 
            f"{self.separator}\n"
            f"Feature selezionate ({len(selected_features)}):\n"
        )
        
        for i, feature in enumerate(selected_features, 1):
            message += f"{i}. {feature}\n"
        
        message += f"{self.separator}\n"
        self.log_message(message)
    
    def log_completion(self):
        """Registra il completamento dell'esecuzione con il tempo impiegato."""
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        
        message = (
            f"\n{self.separator}\n"
            "RIEPILOGO ESECUZIONE\n" + 
            f"{self.separator}\n"
            f"Tempo di esecuzione: {minutes} minuti e {seconds:.2f} secondi\n"
            f"{self.separator}\n"
        )
        
        self.log_message(message)