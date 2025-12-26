from sklearn.tree import DecisionTreeClassifier as SklearnDT
from .BaseClassifier import BaseClassifier

class DecisionTreeClassifier(BaseClassifier):
    """Classificatore Decision Tree."""
    
    def get_model(self):
        """
        Restituisce un'istanza di Decision Tree.
        
        Returns:
            DecisionTreeClassifier: Istanza del modello
        """
        return SklearnDT(random_state=42)