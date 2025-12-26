from sklearn.ensemble import RandomForestClassifier as SklearnRF
from .BaseClassifier import BaseClassifier

class RandomForestClassifier(BaseClassifier):
    """Classificatore Random Forest."""
    
    def get_model(self):
        """
        Restituisce un'istanza di Random Forest.
        
        Returns:
            RandomForestClassifier: Istanza del modello
        """
        return SklearnRF(random_state=42)