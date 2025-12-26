from .BaseSelector import BaseSelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
import shap
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleDenseNN(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units, activation_fn):
        super(SimpleDenseNN, self).__init__()
        layers = []
        
        # Costruisce i layer nascosti
        for i in range(hidden_layers):
            units = hidden_units // (i + 1)
            layers.append(nn.Linear(input_size if i == 0 else hidden_units // i, units))
            layers.append(activation_fn())
            layers.append(nn.Dropout(0.3)) 

        layers.append(nn.Linear(units, 8))
        layers.append(activation_fn())

        # Layer finale con output a una unità e attivazione sigmoid
        layers.append(nn.Linear(8, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Inizializzazione Xavier
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Inizializza i bias a zero
                    
                    
class MLPSelector(BaseSelector):
    """Implementazione di un classificatore basato su reti neurali PyTorch"""
    
    def __init__(self, scaler):
        super().__init__("neural_network", scaler)
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def prepare_data(self, X_train_scaled, y_train, X_test_scaled, y_test):
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train,
                                       dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test,
                                      dtype=torch.float32).view(-1, 1)

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def create_model(self, input_size, hyperparameters):
        
        nl, hidden_units, lr, activation = hyperparameters
        
        # Scelta dell'attivazione
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Attivazione {activation} non supportata")
        
        self.model = SimpleDenseNN(input_size, hidden_layers=nl, hidden_units=hidden_units, activation_fn=act_fn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()
        
        return self.model
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Addestra il modello con early stopping.
        
        Args:
            X_train: Dati di training
            y_train: Label di training
            X_test: Dati di validazione
            y_test: Label di validazione
            
        Returns:
            float: Accuratezza del modello migliore sul validation set
        """
        best_acc = 0.0
        best_model_state = None
        patience = 30  # Numero di epoche senza miglioramento prima di fermarsi
        patience_counter = 0
        max_epochs = 500  # Limite massimo di epoche
        
        # Inizializza i pesi se il modello ha questo metodo
        if hasattr(self.model, '_initialize_weights'):
            self.model._initialize_weights()
        
        # Ciclo di training
        for epoch in range(max_epochs):
            # Training del modello
            self.model.train()
            self.optimizer.zero_grad()
            
            output = self.model(X_train)
            train_loss = self.loss_fn(output, y_train)
            train_loss.backward()
            self.optimizer.step()
            
            # Valutazione sul validation set
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_test)
                val_loss = self.loss_fn(val_output, y_test)
                accuracy = ((val_output.round() == y_test).float().mean()).item()
            
            # Controlla se abbiamo un nuovo modello migliore
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_state = self.model.state_dict().copy()  # Salva lo stato del modello migliore
                patience_counter = 0  # Resetta il contatore di pazienza
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                break
        
        # Carica il miglior modello trovato
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return best_acc
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.detach().numpy()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model_weights.pth'))
    
    def save_data(self, path, data):
        X_train, X_test, y_train, y_test = data
        
        pd.DataFrame(X_train.numpy()).to_csv(os.path.join(path, 'X_train.csv'), index=False)
        pd.DataFrame(y_train.numpy()).to_csv(os.path.join(path, 'y_train.csv'), index=False)
        pd.DataFrame(X_test.numpy()).to_csv(os.path.join(path, 'X_test.csv'), index=False)
        
        y_pred = self.predict(self.scaler.inverse_transform(X_test.numpy()))
        y_test_np = y_test.numpy()
        y_concat = np.concatenate((y_test_np, y_pred), axis=1)
        pd.DataFrame(y_concat, columns=['y_true', 'y_pred']).to_csv(
            os.path.join(path, 'Y_test_pred.csv'), index=False)
        
    def get_shap_values(self, data):
        """
        Calcola i valori SHAP per il modello neural network utilizzando DeepExplainer.
        
        Args:
            data (DataFrame): Dati di input
            
        Returns:
            tuple: (shap0_df, shap1_df) - DataFrame con i valori SHAP per entrambe le classi
        """
        # Converti i dati in tensor per PyTorch
        data_numpy = data.to_numpy(dtype=np.float32)
        X_tensor = torch.from_numpy(data_numpy).float()
        
        # Crea un dataset di background (campione rappresentativo)
        n_background = min(100, len(data))
        rng = np.random.default_rng(SEED)
        background_indices = rng.choice(len(data), n_background, replace=False)
        background = X_tensor[background_indices]
        
        # Imposta il modello in modalità valutazione
        self.model.eval()
        
        # Usa DeepExplainer specifico per reti neurali
        explainer = shap.DeepExplainer(self.model, background)

        # Calcola i valori SHAP
        shap_vals = explainer.shap_values(X_tensor)
        
        # DeepExplainer restituisce una lista di array se il modello ha output multi-classe
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap0, shap1 = shap_vals
        else:
            # Se il modello restituisce solo un logit/probabilità per la classe positiva
            shap1 = shap_vals
            if len(shap1.shape) > 2:
                shap1 = shap1.squeeze(-1)
            shap0 = -shap1  # approssimazione per la classe negativa
        
        # Converti in DataFrames
        shap0_df = pd.DataFrame(shap0, columns=data.columns)
        shap1_df = pd.DataFrame(shap1, columns=data.columns)
        
        return shap0_df, shap1_df