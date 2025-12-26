from abc import ABC, abstractmethod

class BaseSelector(ABC):
    """Classe base astratta per tutti i classificatori"""
    
    def __init__(self, name, scaler):
        self.name = name
        self.scaler = scaler
    
    @abstractmethod
    def prepare_data(self, X, y, test_size=0.2):
        """Prepara i dati per il training e la valutazione"""
        pass
    
    @abstractmethod
    def create_model(self, input_size, hyperparameters):
        """Crea il modello con gli iperparametri specificati"""
        pass
    
    @abstractmethod
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Addestra il modello e valuta le sue prestazioni"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Effettua predizioni su nuovi dati"""
        pass
    
    @abstractmethod
    def save_model(self, path):
        """Salva il modello su disco"""
        pass
    
    @abstractmethod
    def save_data(self, path, data):
        """Salva i dati di training e test"""
        pass
    
    @abstractmethod
    def get_shap_values(self, data):
        """
        Calcola i valori SHAP per il modello.
        
        Args:
            data (DataFrame): Dati di input
            
        Returns:
            DataFrame: DataFrame con i valori SHAP
        """
        pass
