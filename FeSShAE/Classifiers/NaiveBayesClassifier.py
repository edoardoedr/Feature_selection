from sklearn.naive_bayes import GaussianNB
from .BaseClassifier import BaseClassifier

class NaiveBayesClassifier(BaseClassifier):
    """Classificatore Naive Bayes."""
    
    def get_model(self):
        """
        Restituisce un'istanza di Naive Bayes.
        
        Returns:
            GaussianNB: Istanza del modello
        """
        return GaussianNB()