import os
import time
from datetime import datetime

class LoggerFeSShAE:
    """
    Logger per la classe FeSShAE che registra in modo strutturato le informazioni 
    di esecuzione del processo di selezione delle caratteristiche.
    """

    def __init__(self, log_dir):
        """
        Inizializza il logger.
        
        Args:
            log_dir (str): Directory dove salvare il file di log
        """
        self.log_dir = log_dir
        self.start_time = time.time()
        self.separator = "=" * 80
        self.semi_separator = "-" * 50
        
        # Crea la directory di log se non esiste
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log_path = os.path.join(log_dir, "featureshae_log.txt")
        
        # Inizializza il file di log con intestazione
        with open(self.log_path, "w") as f:
            f.write(f"{self.separator}\n")
            f.write(f"FeSShAE - FEATURE SELECTION SYSTEM USING SHAP AND AUTOENCODER\n")
            f.write(f"{self.separator}\n\n")
            f.write(f"Esecuzione iniziata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_iteration_start(self, iteration, total_selected_features=None, selected_features_list=None):
        """
        Registra l'inizio di una nuova iterazione con dettagli.
        
        Args:
            iteration (int): Numero dell'iterazione
            total_selected_features (int, optional): Numero totale di caratteristiche selezionate finora
            selected_features_list (list, optional): Lista delle caratteristiche selezionate finora
        """
        message = (
            f"\n{self.separator}\n"
            f"ITERAZIONE {iteration} - INIZIO\n"
            f"{self.separator}\n"
        )
        
        if total_selected_features is not None:
            message += f"Caratteristiche selezionate finora: {total_selected_features}\n\n"
            
        if selected_features_list and len(selected_features_list) > 0:
            message += "Caratteristiche selezionate in precedenza:\n"
            for i, feature_group in enumerate(selected_features_list):
                message += f"  Iterazione {i}: {feature_group}\n"
            message += "\n"
        
        self.log_message(message)
    
    def log_iteration_end(self, iteration, selected_features=None):
        """
        Registra la fine di un'iterazione con i risultati.
        
        Args:
            iteration (int): Numero dell'iterazione
            selected_features (list, optional): Caratteristiche selezionate in questa iterazione
            metrics (dict, optional): Metriche di performance del modello
        """
        message = (
            f"\n{self.separator}\n"
            f"ITERAZIONE {iteration} - RISULTATI\n"
            f"{self.separator}\n"
        )
        
        if list(selected_features):
            message += f"Caratteristiche selezionate: {selected_features}\n"
            message += f"Numero di caratteristiche: {len(selected_features)}\n\n"
        
        
        message += f"{self.separator}\n"
        self.log_message(message)
    
    def log_filter_start(self, n_clusters):
        """
        Registra l'inizio della fase di autoencoder.
        
        Args:
            n_clusters (int): Numero di cluster da analizzare
        """
        message = (
            f"\nINIZIO FASE FILTRAGGIO\n"
            f"Numero di cluster da analizzare: {n_clusters}\n"
            f"{self.semi_separator}"
        )
        self.log_message(message)
    
    def log_filter_results(self, selected_features):
        """
        Registra i risultati della fase di autoencoder.
        
        Args:
            selected_features (list): Caratteristiche selezionate dall'autoencoder
        """
        if not selected_features:
            message = "\nNessuna caratteristica selezionata dal filter\n"
        else:
            message = (
                f"\nRISULTATI FILTRO\n"
                f"Caratteristiche selezionate: {len(selected_features)}\n"
                f"Lista: {selected_features}\n"
                f"{self.semi_separator}"
            )
        self.log_message(message)
    
    def log_model_search_start(self, classifier_type, features_count):
        """
        Registra l'inizio della ricerca del modello.
        
        Args:
            classifier_type (str): Tipo di classificatore utilizzato
            features_count (int): Numero di caratteristiche usate
        """
        message = (
            f"\nINIZIO RICERCA MODELLO\n"
            f"Classificatore: {classifier_type}\n"
            f"Numero di caratteristiche: {features_count}\n"
            f"{self.semi_separator}"
        )
        self.log_message(message)
    
    def log_model_search_results(self, best_results):
        """
        Registra i risultati della ricerca del modello.
        
        Args:
            best_params (dict): Migliori parametri trovati
            best_score (float): Miglior punteggio ottenuto
        """
        
        best_params = best_results.get('best_hp', {})
        best_acc = best_results.get('best_acc', None)
        best_score = best_results.get('best_scores', None)
        
        message = (
            f"\nRISULTATI RICERCA MODELLO\n"
            f"Miglior punteggio: {best_score}\n"
            f"Miglior accuratezza: {best_acc:.4f}\n"
            f"Migliori parametri: {best_params}\n"
            f"{self.semi_separator}"
        )
        self.log_message(message)
    
    def log_shap_start(self):
        """Registra l'inizio dell'analisi SHAP."""
        message = (
            f"\nINIZIO ANALISI SHAP\n"
            f"{self.semi_separator}"
        )
        self.log_message(message)
        
    
    def log_shap_results(self, selected_features, shap_scores=None):
        """
        Registra i risultati dell'analisi SHAP.
        
        Args:
            selected_features (list/DataFrame): Caratteristiche selezionate tramite SHAP
            shap_scores (DataFrame): DataFrame con indice dei geni e colonne 'corr' e 'intensity'
        """
        message = f"\nRISULTATI ANALISI SHAP\n"
        
        # Gestione selected_features come DataFrame o lista
        if hasattr(selected_features, 'tolist'):
            # È un DataFrame o Series, lo convertiamo in lista
            feature_list = selected_features.tolist() if hasattr(selected_features, 'tolist') else selected_features.values.tolist()
            message += f"Caratteristiche selezionate: {len(feature_list)}\n"
            # Mostra le caratteristiche in gruppi di 5 per migliorare la leggibilità
            for i in range(0, len(feature_list), 5):
                chunk = feature_list[i:i+5]
                message += f"  {', '.join(chunk)}\n"
            message += "\n"
        else:
            # È già una lista
            message += f"Caratteristiche selezionate: {len(selected_features)}\n"
            # Mostra le caratteristiche in gruppi di 5 per migliorare la leggibilità
            for i in range(0, len(selected_features), 5):
                chunk = selected_features[i:i+5]
                message += f"  {', '.join(chunk)}\n"
            message += "\n"
        
        # Gestione shap_scores come DataFrame specifico
        if shap_scores is not None and hasattr(shap_scores, 'sort_values'):
            message += "Punteggi SHAP (ordinati per intensità):\n"
            
            try:
                # Ordina per intensità (valore assoluto dell'importanza)
                sorted_df = shap_scores.sort_values('Intensity', ascending=False)
                
                # Mostra i primi 15 geni con i loro punteggi
                for index, row in sorted_df.head(15).iterrows():
                    corr = row['Correlation']
                    intensity = row['Intensity']
                    feature = row['Feature']
                    # Formatta con segno + esplicito per valori positivi di correlazione
                    corr_sign = f"+{corr:.3f}" if corr > 0 else f"{corr:.3f}"
                    message += f"  {feature}: intensità = {intensity:.3f}, correlazione = {corr_sign}\n"
                
                # Indica se ci sono altri geni non mostrati
                if len(sorted_df) > 15:
                    message += f"  ... e altri {len(sorted_df) - 15} feature\n"
            except Exception as e:
                # Fallback in caso di errori
                message += f"  Errore nell'elaborazione dei punteggi SHAP: {str(e)}\n"
                message += f"  Prime righe del DataFrame:\n{shap_scores.head(5).to_string()}\n"
        
        message += f"{self.semi_separator}"
        self.log_message(message)
    
    def log_final_results(self, selected_features, model_metrics=None):
        """
        Registra i risultati finali dell'intero processo.
        
        Args:
            selected_features (list): Lista finale delle caratteristiche selezionate
            model_metrics (dict, optional): Metriche di performance del modello finale
        """
        message = (
            f"\n{self.separator}\n"
            f"RISULTATI FINALI\n"
            f"{self.separator}\n"
            f"Numero totale di caratteristiche selezionate: {len(selected_features)}\n"
            f"Caratteristiche selezionate: {selected_features}\n\n"
        )
        
        if model_metrics:
            message += "Metriche del modello finale:\n"
            for metric_name, metric_value in model_metrics.items():
                message += f"  {metric_name}: {metric_value}\n"
            message += "\n"
        
        self.log_message(message)
    
    def log_message(self, message):
        """
        Registra un messaggio generico.
        
        Args:
            message (str): Messaggio da registrare
        """
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + '\n')
    
    def log_section(self, title):
        """
        Registra un'intestazione di sezione.
        
        Args:
            title (str): Titolo della sezione
        """
        message = (
            f"\n{'-' * 50}\n"
            f"{title}\n"
            f"{'-' * 50}\n"
        )
        self.log_message(message)
    
    def log_config(self, config):
        """
        Registra la configurazione utilizzata.
        
        Args:
            config (Config): Oggetto configurazione
        """
        message = (
            f"\n{self.separator}\n"
            f"CONFIGURAZIONE\n"
            f"{self.separator}\n"
        )
        
        # Dataset
        message += f"Dataset: {config.dataset.name}\n"
        message += f"Input folder: {config.dataset.input_folder}\n"
        message += f"Label column: {config.dataset.label_column}\n\n"
        
        # Clustering
        message += f"Clustering strategy: {config.clustering.strategy}\n"
        message += f"Clustering parameters: {config.clustering.parameters}\n\n"
        
        # Autoencoder
        message += f"Autoencoder: {config.autoencoder_parameters.AE_name}\n"
        message += f"Learning rate: {config.autoencoder_parameters.lr}\n"
        message += f"Epochs: {config.autoencoder_parameters.epochs}\n\n"
        
        # Search model
        message += f"Classifier: {config.search_model.classifier_type}\n"
        message += f"CV: {config.search_model.cv}\n\n"
        
        # Output
        message += f"Output folder: {config.output.output_folder}\n"
        message += f"N features candidates: {config.n_features_candidates}\n"
        
        message += f"{self.separator}\n"
        self.log_message(message)
    
    def log_completion(self):
        """
        Registra il completamento dell'esecuzione con il tempo impiegato.
        """
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        
        message = (
            f"\n{self.separator}\n"
            f"ESECUZIONE COMPLETATA\n"
            f"{self.separator}\n"
            f"Data e ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Tempo di esecuzione: {hours}h {minutes}m {seconds:.2f}s\n"
            f"{self.separator}\n"
        )
        
        self.log_message(message)