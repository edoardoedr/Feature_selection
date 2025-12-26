from sklearn.dummy import DummyClassifier
from .BaseClassifier import BaseClassifier

class ZeroRClassifier(BaseClassifier):
    """Classificatore ZeroR (baseline)."""
    
    def __init__(self, param_space=None, cv=3, output_dir=None):
        """
        Inizializza il classificatore ZeroR.
        
        Args:
            param_space (dict): Spazio dei parametri (ignorato per ZeroR)
            cv (int): Numero di fold per cross-validation
            output_dir (str): Directory per salvare i risultati
        """
        # ZeroR non ha parametri da ottimizzare, quindi passiamo un dizionario vuoto
        super().__init__({}, cv, output_dir)
    
    def get_model(self):
        """
        Restituisce un'istanza di ZeroR (implementato come DummyClassifier).
        
        Returns:
            DummyClassifier: Istanza del modello
        """
        return DummyClassifier(strategy='most_frequent')