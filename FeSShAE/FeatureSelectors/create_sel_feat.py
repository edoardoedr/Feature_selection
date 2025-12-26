from .SelFeatBy import SHAPFeat, RandomFeat

def get_sel_feat_by(selector_type, output_dir=None, logger=None, shap_sampling=100):
    """
    Factory function che restituisce un selettore di feature in base al tipo specificato.
    
    Args:
        selector_type (str): Tipo di selettore ('SHAP', 'random')
        output_dir (str): Directory dove salvare i risultati
        logger: Logger per registrare informazioni
        shap_sampling (int): Numero di campioni da utilizzare per l'analisi SHAP
        
    Returns:
        object: Istanza del selettore di feature richiesto
        
    Raises:
        ValueError: Se il tipo di selettore non è supportato
    """
    selector_type = selector_type.lower() if isinstance(selector_type, str) else selector_type
    
    if selector_type == 'shap':
        if logger:
            logger.log_message(f"Inizializzazione selettore SHAP con sampling={shap_sampling}")
        return SHAPFeat(output_dir, logger, shap_sampling)
    elif selector_type == 'random':
        if logger:
            logger.log_message("Inizializzazione selettore Random")
        return RandomFeat(output_dir, logger)
    else:
        available_selectors = ['shap', 'random']
        error_msg = f"Tipo di selettore '{selector_type}' non supportato. Selettori disponibili: {', '.join(available_selectors)}"
        if logger:
            logger.log_message(f"ERRORE: {error_msg}")
        raise ValueError(error_msg)