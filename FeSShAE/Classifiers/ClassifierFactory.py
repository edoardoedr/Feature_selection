from .RandomForestClassifier import RandomForestClassifier
from .SVMClassifier import SVMClassifier
from .DecisionTreeClassifier import DecisionTreeClassifier
from .NaiveBayesClassifier import NaiveBayesClassifier
from .ZeroRClassifier import ZeroRClassifier

def create_classifier(classifier_type, param_space, scaler, cv=3, output_dir=None, logger=None):
    """
    Factory per creare l'istanza del classificatore richiesto.
    
    Args:
        classifier_type (str): Tipo di classificatore
        param_space (dict): Spazio dei parametri per GridSearch
        cv (int): Numero di fold per cross-validation
        output_dir (str): Directory per salvare i risultati
        
    Returns:
        BaseClassifier: Istanza del classificatore richiesto
        
    Raises:
        ValueError: Se il tipo di classificatore non è supportato
    """
    classifier_map = {
        'random_forest': RandomForestClassifier,
        'support_vector_machine': SVMClassifier,
        'decision_tree': DecisionTreeClassifier,
        'NaiveBayes': NaiveBayesClassifier,
        'ZeroR': ZeroRClassifier
    }
    
    if classifier_type not in classifier_map:
        raise ValueError(f"Tipo di classificatore non supportato: {classifier_type}. "
                         f"Tipi supportati: {list(classifier_map.keys())}")
    
    # Per ZeroR ignoriamo i parametri
    if classifier_type == 'ZeroR':
        return classifier_map[classifier_type](cv=cv, output_dir=output_dir, scaler=scaler)

    return classifier_map[classifier_type](param_space, cv, output_dir, scaler, logger=logger)