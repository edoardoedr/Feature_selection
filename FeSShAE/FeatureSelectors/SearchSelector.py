import os
import itertools
import numpy as np
import pandas as pd
import scipy.stats as st
from .loggers import SearchSelectorLogger
from .Selectors import MLPSelector, XGBoostSelector, RandomForestSelector, SVMSelector
from sklearn.model_selection import StratifiedKFold
import random
import copy

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class SearchSelector:
    """
    Classe per la ricerca del miglior modello di classificazione
    attraverso la valutazione di diversi iperparametri.
    """
    
    def __init__(self, param_space, iters, selector_name, scaler, output_dir, cv=3):
        self.param_space = param_space
        self.iters = iters
        self.cv = cv
        self.output_dir = output_dir
        self.scaler = scaler
        self.logger = SearchSelectorLogger(iteration=iters, output_dir=self.output_dir)
        self.selector_registry = {
            'neural_network': MLPSelector,
            'xgboost': XGBoostSelector,
            'random_forest': RandomForestSelector,
            'svm': SVMSelector
        }
        self.selector_name = selector_name
    
    def get_selector(self, selector_type):
        """
        Ottiene un'istanza del classificatore specificato.
        
        Args:
            selector_type (str): Il tipo di classificatore da utilizzare
                                  ('neural_network', 'xgboost', 'random_forest', 'svm')
        
        Returns:
            Baseselector: Un'istanza del classificatore richiesto
        """
        if selector_type not in self.selector_registry:
            raise ValueError(f"Classificatore '{selector_type}' non supportato. "
                            f"Tipi supportati: {list(self.selector_registry.keys())}")
        
        return self.selector_registry[selector_type](scaler = self.scaler)
    
    def write_results(self, selector, best_hp, best_acc, best_data, scores, iteration):
        """
        Scrive i risultati della ricerca su disco.
        
        Args:
            selector (Baseselector): Il classificatore migliore
            best_hp (tuple): Gli iperparametri migliori
            best_acc (float): La miglior accuratezza
            best_data (tuple): I dati utilizzati per il miglior modello
            scores (list): I punteggi di cross-validation
            iteration (int/str): L'iterazione corrente
        """
        best_model_path = os.path.join(self.output_dir, f'output_iteration_{iteration}')
        os.makedirs(best_model_path, exist_ok=True)
        
        # Salva il modello
        selector.save_model(best_model_path)
        
        # Salva i dati
        selector.save_data(best_model_path, best_data)
        
        # Scrivi le statistiche delle prestazioni del modello
        ci = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))
        
        # Aggiungi queste informazioni al log
        self.logger.log_best_model(
            best_model_idx=self.best_model_idx, 
            best_hyperparams=best_hp, 
            best_acc=best_acc, 
            best_model_type=selector.name,
            best_scores=scores,
            ci_lower=ci[0],
            ci_upper=ci[1]
        )
        
        pd.DataFrame([[best_acc, ci[0], ci[1], str(best_hp), str([round(num, 4) for num in scores])]],
                    columns=['best_accuracy(test)', 'CI95% lower', 'CI95% upper', 'hyperparameters', 'cv_scores']) \
            .to_csv(os.path.join(best_model_path, 'best_model_scores.csv'), index=False)
        
        print(f"Results saved in {best_model_path}")
    
    def search_model(self, X, y):
        """
        Cerca il miglior modello per i dati forniti.
        
        Args:
            X (DataFrame): Features
            y (Series): Target
            selector_type (str): Tipo di classificatore da utilizzare
        
        Returns:
            tuple: (miglior modello, scaler)
        """
        param_space = list(self.param_space.values())
        total_models = np.prod([len(p) for p in param_space])
        
        np.random.seed(SEED)
        random.seed(SEED)
        
        self.logger.log_message(f"Inizia la ricerca del miglior modello - Tipo: {self.selector_name}")
        self.logger.log_message(f"Totale modelli da valutare: {total_models}")
        self.logger.log_message(f"Cross-validation folds: {self.cv}\n")
        
        best_scores = None
        best_selector = None
        best_hp = None
        best_acc = -np.inf
        best_data = None
        self.best_model_idx = 0
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=SEED)
        cv_splits = list(skf.split(X, y))
        
        i = 0
        for hpset in itertools.product(*param_space):
            i += 1
            
            scores = []
            best_selector_cv = None
            best_acc_cv = -np.inf
            best_data_cv = None
            
            self.logger.log_message(f"\nModello {i}/{total_models} - Tipo: {self.selector_name}")
            self.logger.log_message(f"Iperparametri: {hpset}")
            
            for cv_idx, (train_idx, test_idx) in enumerate(cv_splits):

                # Crea una nuova istanza del classificatore per ogni CV
                selector = self.get_selector(self.selector_name)
                
                # Prepara i dati
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Effettua lo scaling dei dati
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                data = selector.prepare_data(X_train_scaled, y_train, X_test_scaled, y_test)
                X_train, X_test, y_train, y_test = data
                
                # Crea e addestra il modello
                selector.create_model(X.shape[1], hpset)
                acc = selector.train_and_evaluate(X_train, y_train, X_test, y_test)
                
                # Log del risultato della fold
                self.logger.log_cv_fold(cv_idx + 1, self.cv, acc)

                if acc > best_acc_cv:
                    best_acc_cv = acc
                    best_data_cv = copy.deepcopy(data)
                    best_selector_cv = copy.deepcopy(selector)
                
                scores.append(acc)
            
            mean_acc = np.mean(scores)
            
            # Log del risultato del modello
            self.logger.log_model_evaluation(
                model_idx=i,
                total_models=total_models,
                hyperparams=hpset,
                model_type=self.selector_name,
                scores=scores,
                mean_acc=mean_acc
            )
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_selector = copy.deepcopy(best_selector_cv)
                best_hp = hpset
                best_data = copy.deepcopy(best_data_cv)
                best_scores = scores
                self.best_model_idx = i
        
        self.write_results(best_selector, best_hp, best_acc, best_data, best_scores, self.iters)
        self.logger.log_completion()
        
        return best_selector, {'best_hp': best_hp, 'best_acc': best_acc, 'best_scores': best_scores}