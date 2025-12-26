from .DataPreparation import DataPreparationFactory
from .tools import Config, LoggerFeSShAE, get_scaler
from .Filters import create_filter
from .FeatureSelectors import SearchSelector, get_sel_feat_by
from .Classifiers import create_classifier
import os
import pandas as pd


class FeSShAE:
    def __init__(self, config_path):
        """Inizializza il selettore di caratteristiche con il file di configurazione.
        
        Args:
            config_path: Percorso del file di configurazione YAML
        """
        # Caricamento configurazione
        self.config = Config(config_path)
        
        # Inizializzazione logger
        self.logger = LoggerFeSShAE(self.config.output.output_folder)
        self.logger.log_message("Inizializzazione FeSShAE")
        
        # Registra la configurazione
        self.logger.log_config(self.config)
        
        self.scaler = get_scaler(self.config.scaler)
        
        # Caricamento dati
        self.datas = self._load_data_selection()
        
        # Inizializzazione lista per salvare le feature selezionate
        self.shap_winners = []
        self.sel_feat = get_sel_feat_by(self.config.sel_feat_by,
                                     output_dir=self.config.output.output_folder,
                                     logger=self.logger,
                                     shap_sampling=self.config.shap_sampling)
        
        self.iteration = 0
        
    def _load_data_selection(self):
        """Carica i dati per la selezione delle caratteristiche.
        
        Returns:
            dict: Dati preparati per la selezione delle caratteristiche
        """
        self.logger.log_section("CARICAMENTO DATI")
        
        data_preparation = DataPreparationFactory(
            dataset_name=self.config.dataset.name,
            input_folder=self.config.dataset.input_folder,
            output_folder=self.config.output.output_folder,
            cluster_strategy=self.config.clustering.strategy,
            cluster_parameters=self.config.clustering.parameters,
            cluster_on_correlation=self.config.clustering.cluster_on_correlation,
            label_column=self.config.dataset.label_column
        )
        data = data_preparation.export_data_clusters()
        # Log delle dimensioni dei dati
        features_count = len(data['features'])
        samples_count = data['data'].shape[0]
        clusters_count = max(data['clusters_over_features']['Cluster']) + 1
        
        self.logger.log_message(f"Dati caricati con successo:")
        self.logger.log_message(f"  Campioni: {samples_count}")
        self.logger.log_message(f"  Caratteristiche: {features_count}")
        self.logger.log_message(f"  Cluster: {clusters_count-1}")
        
        return data
    
    def run_iteration(self):
        """Esegue un'iterazione del processo di selezione delle caratteristiche.
        """
        # Log inizio iterazione
        total_selected = sum([len(l) for l in self.shap_winners])
        self.logger.log_iteration_start(self.iteration, total_selected, self.shap_winners)
                
        # Preparazione dati per il filtro AE
        features = self.datas['features'].copy()
        features_data = self.datas['features_data'][features].copy()
        
        # Log dell'inizio della fase autoencoder
        n_clusters = max(self.datas['clusters_over_features']['Cluster']) + 1
        self.logger.log_filter_start(n_clusters-1)
        
        # Applicazione del filtro AE usando la nuova struttura config
        FeatureFilter = create_filter(self.config.filters, self.iteration, self.config.output.output_folder, self.scaler)
        
        selected_features = FeatureFilter.filter(features_data, self.datas['clusters_over_features'])
        
        # Log dei risultati dell'autoencoder
        self.logger.log_filter_results(selected_features)
        
        
        if selected_features:
            
            # Applicazione del modello di ricerca
            current_data = self.datas['data'][selected_features].copy()
            labels = self.datas['labels']
            
            # Configurazione del modello di ricerca
            selector_type = self.config.search_selector.selector_type
            
            # Log dell'inizio della ricerca del modello
            self.logger.log_model_search_start(selector_type, len(selected_features))
            
            search_selector = SearchSelector(
                param_space=self.config.search_selector.param_space[selector_type],
                iters=self.iteration,
                selector_name=selector_type,
                output_dir=self.config.output.output_folder,
                cv=self.config.search_selector.cv
            )
            
            model, results = search_selector.search_selector(current_data, labels)
            # Log dei risultati della ricerca del modello
            self.logger.log_model_search_results(results)
            
            # Log dell'inizio della selezione con SHAP
            self.logger.log_shap_start()
            
            shap_selected, shap_scores = self.sel_feat.select_features(
                model=model,
                data=current_data,
                scaler=self.scaler,
                iteration=self.iteration,
            )
            
            # Log dei risultati SHAP
            self.logger.log_shap_results(shap_selected, shap_scores)
            
            self.shap_winners.append(shap_selected)
            self.datas['features'] = list(set(features) - set(shap_selected))
            self.datas['clusters_over_features'] = self.datas['clusters_over_features'][~self.datas['clusters_over_features']['Feature'].isin(shap_selected)].copy()

            # Log dei risultati dell'iterazione
            self.logger.log_iteration_end(self.iteration, shap_selected)
            
            self.iteration += 1

    def run(self):
        
        """Esegue il processo di selezione delle caratteristiche."""
        self.logger.log_section("AVVIO DEL PROCESSO DI SELEZIONE DELLE CARATTERISTICHE")
        self.logger.log_message(f"Target di caratteristiche da selezionare: {self.config.n_features_candidates}")
        
        while sum(len(l) for l in self.shap_winners) < self.config.n_features_candidates:
            self.run_iteration()
        
        repeated_winner_features = [item for sublist in self.shap_winners for item in sublist]
        repeated_winner_features = repeated_winner_features[:self.config.n_features_candidates]
        pd.DataFrame(repeated_winner_features, columns=['features']).to_excel(os.path.join(self.config.output.output_folder, 'repeated_winner.xlsx'), index=False)
        
        self.logger.log_section("FASE FINALE - VALUTAZIONE DEL MODELLO COMPLETO")
        
        #features = pd.read_excel(os.path.join(self.config.output.output_folder, 'repeated_winner.xlsx'))['features'].values
        final_data = self.datas['data'][repeated_winner_features].copy()
        
        classifier_type = self.config.classifier.classifier_type
        # Log dell'inizio della ricerca del modello
        self.logger.log_message(f"Valutazione finale su {len(repeated_winner_features)} caratteristiche selezionate: {repeated_winner_features} usando il modello {classifier_type}")
        
        classifier = create_classifier(self.config.classifier_type,
                                       self.config.classifier.param_space[classifier_type],
                                       self.scaler,
                                       self.config.classifier.cv,
                                       self.config.output.output_folder,
                                       logger=self.logger)
        
        results = classifier.evaluate(final_data, self.datas['labels'])
        
        # Log del completamento
        self.logger.log_completion()
        