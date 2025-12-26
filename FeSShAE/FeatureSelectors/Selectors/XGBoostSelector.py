import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import shap
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
from .BaseSelector import BaseSelector

class XGBoostSelector(BaseSelector):
    """Implementazione di un classificatore XGBoost"""
    
    def __init__(self, scaler):
        super().__init__("xgboost", scaler)
        self.model = None
    
    def prepare_data(self, X_train_scaled, y_train, X_test_scaled, y_test):
        return X_train_scaled, y_train, X_test_scaled, y_test

    def create_model(self, input_size, hyperparameters):
        n_estimators, learning_rate, max_depth, colsample_bytree = hyperparameters
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            colsample_bytree=colsample_bytree,
            eval_metric='logloss',
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
        Calcola i valori SHAP per il modello XGBoost.
        
        Args:
            data (DataFrame): Dati di input
            
        Returns:
            DataFrame: DataFrame con i valori SHAP
        """
        # Per XGBoost possiamo usare TreeExplainer che è più efficiente
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(data)
            
        # Unpack for binary
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap0, shap1 = shap_values
        else:
            shap1 = shap_values[1]
            shap0 = -shap1
            
        shap0_df = pd.DataFrame(shap0, columns=data.columns)
        shap1_df = pd.DataFrame(shap1, columns=data.columns)
        
        return shap0_df, shap1_df
