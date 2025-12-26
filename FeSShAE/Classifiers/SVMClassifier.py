from sklearn.svm import SVC
from .BaseClassifier import BaseClassifier

class SVMClassifier(BaseClassifier):
    """Classificatore Support Vector Machine."""
    
    def get_model(self):
        """
        Restituisce un'istanza di SVM.
        
        Returns:
            SVC: Istanza del modello
        """
        return SVC(probability=True, random_state=42)