from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Impostazione del seed globale
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class BaseClassifier(ABC):
    """Classe base astratta per tutti i classificatori."""
    
    def __init__(self, param_space, scaler, cv=3, output_dir=None, logger=None):
        """
        Inizializza il classificatore base.
        
        Args:
            param_space (dict): Spazio dei parametri per GridSearch
            cv (int): Numero di fold per cross-validation
            output_dir (str): Directory per salvare i risultati
            logger (LoggerFeSShAE, optional): Logger per registrare le operazioni
        """
        self.param_space = param_space
        self.cv = cv
        self.output_dir = output_dir
        self.best_model = None
        self.best_params = None
        self.scaler = scaler
        self.logger = logger
        
    @abstractmethod
    def get_model(self):
        """
        Restituisce l'istanza del modello specifico.
        
        Returns:
            object: Istanza del modello
        """
        pass
        
    def evaluate(self, X, y):
        """
        Esegue una valutazione completa del modello utilizzando GridSearch e StratifiedKFold.
        
        Args:
            X (DataFrame): Dati di input
            y (Series): Target labels
            
        Returns:
            dict: Metriche di valutazione
        """
        model_name = self.__class__.__name__
        if self.logger:
            self.logger.log_section(f"VALUTAZIONE CLASSIFICATORE {model_name}")
            self.logger.log_message(f"Inizializzazione valutazione con {self.cv}-fold cross-validation")
            self.logger.log_message(f"Dati di input: {X.shape[0]} campioni, {X.shape[1]} caratteristiche")
            self.logger.log_message(f"Spazio dei parametri: {self.param_space}")
        
        # Preparazione del cross-validator
        stratified_kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=SEED)
        
        # Preparazione del GridSearch
        model = self.get_model()
        if self.logger:
            self.logger.log_message(f"Avvio GridSearchCV per ottimizzazione parametri")
        
        grid_search = GridSearchCV(
            model,
            self.param_space,
            cv=stratified_kfold,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
        )
        
        # Esecuzione della GridSearch
        grid_search.fit(X, y)
        
        # Estrazione del miglior modello
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        if self.logger:
            self.logger.log_message(f"GridSearch completata. Migliori parametri trovati: {self.best_params}")
            self.logger.log_message(f"Miglior score di GridSearch: {grid_search.best_score_:.4f}")
            self.logger.log_message("Avvio valutazione dettagliata con cross-validation")
        
        # Valutazione completa con le metriche
        metrics = self._evaluate_with_cv(X, y, stratified_kfold)
        
        # Salvataggio dei risultati
        self._save_results(metrics)
        
        if self.logger:
            metrics_log = "\nRISULTATI VALUTAZIONE\n"
            metrics_log += f"Accuratezza media: {metrics['accuracy']:.4f}\n"
            metrics_log += f"Precisione media: {metrics['precision']:.4f}\n"
            metrics_log += f"Recall media: {metrics['recall']:.4f}\n"
            metrics_log += f"F1-Score media: {metrics['f1_score']:.4f}\n"
            if 'auc' in metrics:
                metrics_log += f"AUC media: {metrics['auc']:.4f}\n"
            self.logger.log_message(metrics_log)
        
        return metrics
    
    def _evaluate_with_cv(self, X, y, cv):
        """
        Valuta il modello utilizzando tutte le metriche di interesse con CV.
        
        Args:
            X (DataFrame): Dati di input
            y (Series): Target labels
            cv: Cross-validator
            
        Returns:
            dict: Metriche di valutazione
        """
        # Liste per raccogliere i risultati delle metriche
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        
        # Log inizio valutazione
        if self.logger:
            self.logger.log_message(f"Inizio valutazione dettagliata con {self.cv} fold")
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if self.logger:
                self.logger.log_message(f"Fold {fold+1}/{self.cv} - Training set: {X_train.shape[0]} campioni, Test set: {X_test.shape[0]} campioni")
            
            # Standardizzazione dei dati
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Addestramento con i migliori parametri
            model = self.get_model().set_params(**self.best_params)
            model.fit(X_train_scaled, y_train)
            
            # Predizione
            y_pred = model.predict(X_test_scaled)
            
            # Per AUC servono le probabilità
            try:
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                aucs.append(auc)
                if self.logger:
                    self.logger.log_message(f"Fold {fold+1} - AUC: {auc:.4f}")
            except (AttributeError, IndexError):
                aucs.append(np.nan)  # Alcuni modelli potrebbero non supportare predict_proba
                if self.logger:
                    self.logger.log_message(f"Fold {fold+1} - AUC: non disponibile")
            
            # Calcolo delle metriche
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
            
            if self.logger:
                fold_metrics = f"Fold {fold+1} - Accuratezza: {acc:.4f}, Precisione: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
                self.logger.log_message(fold_metrics)
        
        # Calcolo delle medie
        metrics = {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'best_params': self.best_params,
            'fold_accuracies': accuracies,
            'fold_precisions': precisions,
            'fold_recalls': recalls,
            'fold_f1_scores': f1_scores
        }
        
        if not all(np.isnan(aucs)):
            metrics['auc'] = np.mean([x for x in aucs if not np.isnan(x)])
            metrics['fold_aucs'] = aucs
            
            if self.logger:
                aucs_str = ', '.join([f"{auc:.4f}" if not np.isnan(auc) else "N/A" for auc in aucs])
                self.logger.log_message(f"AUC per fold: [{aucs_str}]")
        
        if self.logger:
            acc_str = ', '.join([f"{acc:.4f}" for acc in accuracies])
            self.logger.log_message(f"Accuratezze per fold: [{acc_str}]")
            self.logger.log_message(f"Deviazione standard accuratezza: {np.std(accuracies):.4f}")
        
        return metrics
    
    def _save_results(self, metrics):
        """
        Salva i risultati della valutazione.
        
        Args:
            metrics (dict): Metriche da salvare
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Preparazione dataframe risultati
            class_name = self.__class__.__name__
            results_df = pd.DataFrame({
                'classifier': [class_name],
                'accuracy': [metrics['accuracy']],
                'precision': [metrics['precision']],
                'recall': [metrics['recall']],
                'f1_score': [metrics['f1_score']]
            })
            
            if 'auc' in metrics:
                results_df['auc'] = metrics['auc']
            
            # Preparazione dataframe parametri
            params_df = pd.DataFrame([metrics['best_params']])
            params_df['classifier'] = class_name
            
            # Preparazione dettagli per fold
            fold_details = pd.DataFrame({
                'fold': list(range(1, len(metrics['fold_accuracies'])+1)),
                'accuracy': metrics['fold_accuracies'],
                'precision': metrics['fold_precisions'],
                'recall': metrics['fold_recalls'],
                'f1_score': metrics['fold_f1_scores']
            })
            if 'fold_aucs' in metrics:
                fold_details['auc'] = metrics['fold_aucs']
            fold_details['classifier'] = class_name
            
            # Salvataggio su file
            results_path = os.path.join(self.output_dir, 'classifier_results.csv')
            params_path = os.path.join(self.output_dir, 'classifier_best_params.csv')
            fold_path = os.path.join(self.output_dir, 'classifier_fold_details.csv')
            
            # Append se il file esiste, altrimenti crea nuovo
            if os.path.exists(results_path):
                results_df.to_csv(results_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(results_path, index=False)
                
            if os.path.exists(params_path):
                params_df.to_csv(params_path, mode='a', header=False, index=False)
            else:
                params_df.to_csv(params_path, index=False)
                
            if os.path.exists(fold_path):
                fold_details.to_csv(fold_path, mode='a', header=False, index=False)
            else:
                fold_details.to_csv(fold_path, index=False)
            
            if self.logger:
                self.logger.log_message(f"Risultati salvati nella directory: {self.output_dir}")
                self.logger.log_message(f"File salvati: classifier_results.csv, classifier_best_params.csv, classifier_fold_details.csv")
