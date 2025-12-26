from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import pickle
import shap
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
from .BaseSelector import BaseSelector

class SVMSelector(BaseSelector):
    """Implementazione di un classificatore SVM"""
    
    def __init__(self, scaler):
        super().__init__("svm", scaler)
        self.model = None
    
    def prepare_data(self, X_train_scaled, y_train, X_test_scaled, y_test):
        return X_train_scaled, y_train, X_test_scaled, y_test

    def create_model(self, input_size, hyperparameters):
        C, kernel, gamma = hyperparameters

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=SEED
        )
        
        return self.model
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
    
    def save_model(self, path):
        with open(os.path.join(path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
    
    def save_data(self, path, data):
        X_train, X_test, y_train, y_test = data
        
        pd.DataFrame(X_train).to_csv(os.path.join(path, 'X_train.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(path, 'y_train.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(path, 'X_test.csv'), index=False)
        
        y_pred = self.predict(self.scaler.inverse_transform(X_test))
        y_test_np = np.array(y_test).reshape(-1, 1)
        y_concat = np.concatenate((y_test_np, y_pred), axis=1)
        pd.DataFrame(y_concat, columns=['y_true', 'y_pred']).to_csv(
            os.path.join(path, 'Y_test_pred.csv'), index=False)
        
    def get_shap_values(self, data):
        """
        Calcola i valori SHAP per il modello SVM.
        
        Args:
            data (DataFrame): Dati di input
            
        Returns:
            tuple: (shap0_df, shap1_df) - DataFrame con i valori SHAP per entrambe le classi
        """
        # Per SVM è meglio usare KernelExplainer
        # Crea un sottoinsieme dei dati per il background
        background = shap.sample(data, 100, random_state=SEED)
        
        # Funzione di predizione
        def f(x):
            return self.model.predict_proba(x)
        
        # Crea l'explainer
        explainer = shap.KernelExplainer(f, background)
        
        # Calcola i valori SHAP
        shap_values = explainer.shap_values(data)
        
        # Unpack per modello binario
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap0, shap1 = shap_values
        else:
            # Se shap_values è un singolo array, assumiamo che sia per la classe positiva
            shap1 = shap_values
            shap0 = -shap1  # approssimiamo i valori per la classe negativa
        
        # Converti in DataFrame
        shap0_df = pd.DataFrame(shap0, columns=data.columns)
        shap1_df = pd.DataFrame(shap1, columns=data.columns)
        
        return shap0_df, shap1_df