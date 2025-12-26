import yaml
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    name: str
    input_folder: str
    label_column: str

@dataclass
class OutputConfig:
    output_folder: str
    logs_folder: str

class ClusteringConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.strategy = config_dict.get('strategy', 'KMeans')
        self._parameters = config_dict.get('parameters', "searching")
    
    @property
    def parameters(self) -> Union[str, Dict[str, Any]]:
        """
        Restituisce i parametri di clustering che possono essere:
        - La stringa "searching"
        - Un dizionario di parametri
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: Union[str, Dict[str, Any]]):
        """
        Imposta i parametri di clustering.
        
        Args:
            value: Può essere la stringa "searching" o un dizionario di parametri
        """
        if not isinstance(value, (str, dict)):
            raise TypeError("I parametri di clustering devono essere una stringa o un dizionario")
        
        if isinstance(value, str) and value != "searching":
            raise ValueError('Se i parametri sono una stringa, deve essere "searching"')
            
        self._parameters = value

@dataclass
class AutoencoderConfig:
    lr: float = 0.001
    epochs: int = 100
    AE_name: str = 'LinearAE'

@dataclass
class MedoidConfig:
    distance_metric: str = 'euclidean'

@dataclass
class FiltersConfig:
    filter_type: str = 'AE'
    autoencoder_parametrs: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    medoid_parameters: MedoidConfig = field(default_factory=MedoidConfig)


class SearchSelectorConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config_dict = config_dict
        self.selector_type = config_dict.get('selector_type', 'neural_network')
        self.cv = config_dict.get('cv', 3)
        self._param_space = config_dict.get('param_space', {})
        
    @property
    def param_space(self) -> Dict[str, Any]:
        """Ritorna il dizionario param_space completo."""
        return self._param_space

class ClassifierConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config_dict = config_dict
        self.selector_type = config_dict.get('classifier_type', 'random_forest')
        self.cv = config_dict.get('cv', 3)
        self.scaler = config_dict.get('scaler', 'standard')
        self._param_space = config_dict.get('param_space', {})
        
    @property
    def param_space(self) -> Dict[str, Any]:
        """Ritorna il dizionario param_space completo."""
        return self._param_space

class Config:
    def __init__(self, config_path: str):
        """
        Inizializza l'oggetto Config leggendo il file YAML.
        
        Args:
            config_path: Percorso del file di configurazione YAML
        """
        self._config_dict = self._load_config(config_path)
        
        # Inizializzazione delle sezioni principali
        self.dataset = DatasetConfig(**self._config_dict.get('dataset', {}))
        self.output = OutputConfig(**self._config_dict.get('output', {}))
        
        # Per clustering utilizziamo una classe personalizzata che gestisce sia "searching" che dizionari
        clustering_dict = self._config_dict.get('clustering', {})
        self.clustering = ClusteringConfig(clustering_dict)
        
        self.n_features_candidates = self._config_dict.get('n_features_candidates', 10)
        self.shap_sampling = self._config_dict.get('shap_sampling', 100)
        
        filters_dict = self._config_dict.get('filters', {})
        self.filters = FiltersConfig(**filters_dict)
        
        self.search_selector = SearchSelectorConfig(self._config_dict.get('search_selector', {}))
        self.classifier = ClassifierConfig(self._config_dict.get('search_classifier', {}))
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carica il file di configurazione YAML.
        
        Args:
            config_path: Percorso del file YAML
            
        Returns:
            Dizionario di configurazione
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Errore nel caricamento del file di configurazione: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte la configurazione in un dizionario.
        
        Returns:
            Dizionario della configurazione completa
        """
        return self._config_dict

    def save(self, path: str) -> None:
        """
        Salva la configurazione in un file YAML.
        
        Args:
            path: Percorso dove salvare il file
        """
        with open(path, 'w') as file:
            yaml.dump(self._config_dict, file, default_flow_style=False)
            
    def validate(self) -> bool:
        """
        Valida che la configurazione contenga tutti i parametri necessari
        e che siano di tipo corretto.
        
        Returns:
            True se la configurazione è valida
            
        Raises:
            ValueError: Se la configurazione non è valida
        """
        # Verifica i valori obbligatori
        if not hasattr(self.dataset, 'input_folder') or not self.dataset.input_folder:
            raise ValueError("Manca il parametro obbligatorio 'input_folder' in dataset")
            
        if not hasattr(self.output, 'output_folder') or not self.output.output_folder:
            raise ValueError("Manca il parametro obbligatorio 'output_folder' in output")
            
        # Verifica la strategia di clustering
        valid_strategies = ['KMeans', 'hierarchical', 'DBScan', 'KMedoids']
        if self.clustering.strategy not in valid_strategies:
            raise ValueError(f"La strategia di clustering '{self.clustering.strategy}' non è valida. "
                            f"Deve essere una tra: {', '.join(valid_strategies)}")
        
        return True